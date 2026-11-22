"""Execute one leased attempt: build the agent, run it in a fresh environment, heartbeat, finalize."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping

from sqlalchemy.orm import Session, sessionmaker

from app.accounting import UsageSummary
from app.agents.registry import AdapterFactory, AgentConfigSpec, build_agent
from app.db.models import AgentConfig, Run, TaskVersion
from app.runner import run_task_attempt
from app.tasks import AgentTaskView, EvaluatorTaskView
from app.worker.queue import Lease, LeaseLost, cancel_requested, fail_attempt, finalize_attempt, heartbeat

log = logging.getLogger(__name__)


class _Heartbeat:
    """Renews the lease in the background; sets ``lost`` the moment it is no longer ours."""

    def __init__(self, sm: sessionmaker[Session], lease: Lease, *, lease_seconds: float, interval: float) -> None:
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(sm, lease, lease_seconds, interval), name=f"heartbeat-{lease.job_id}", daemon=True
        )

    def __enter__(self) -> _Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self, sm: sessionmaker[Session], lease: Lease, lease_seconds: float, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                if not heartbeat(sm, lease, lease_seconds=lease_seconds):
                    log.warning("lost lease on job %s; stopping attempt %s", lease.job_id, lease.attempt_id)
                    self.lost.set()
                    return
            except Exception:  # noqa: BLE001 - a transient DB error must not kill the attempt; the lease may survive
                log.exception("heartbeat failed for job %s", lease.job_id)


def _spec(config: AgentConfig) -> AgentConfigSpec:
    return AgentConfigSpec(
        name=config.name,
        adapter=config.adapter,
        model=config.model,
        prompt=config.prompt,
        tools=tuple(config.tools) if config.tools is not None else None,
        limits=config.limits,
        settings=config.settings,
    )


def execute(
    sm: sessionmaker[Session],
    lease: Lease,
    *,
    adapters: Mapping[str, AdapterFactory],
    lease_seconds: float,
    heartbeat_seconds: float,
    retry_backoff_seconds: float,
) -> None:
    """Never raises for agent or adapter problems: every outcome ends as a recorded attempt."""
    try:
        with sm() as session:
            run = session.get(Run, lease.run_id)
            task = session.get(TaskVersion, run.task_version_id)
            spec = _spec(session.get(AgentConfig, run.agent_config_id))
            agent_view = AgentTaskView.model_validate(task.agent_view)
            evaluator_view = EvaluatorTaskView.model_validate(task.evaluator_view)
    except Exception as exc:  # noqa: BLE001 - unreadable inputs are an infrastructure error, retried
        log.exception("could not load inputs for attempt %s", lease.attempt_id)
        _fail(sm, lease, "WORKER_EXCEPTION", f"{type(exc).__name__}: {exc}", True, retry_backoff_seconds)
        return

    try:
        agent = build_agent(spec, adapters)
    except Exception as exc:  # noqa: BLE001 - an unusable configuration will not fix itself; do not retry
        _fail(sm, lease, "ADAPTER_UNAVAILABLE", f"{type(exc).__name__}: {exc}", False, retry_backoff_seconds)
        return

    started = time.monotonic()
    crash: Exception | None = None
    with _Heartbeat(sm, lease, lease_seconds=lease_seconds, interval=heartbeat_seconds) as beat:
        try:
            result = run_task_attempt(
                agent_view,
                evaluator_view,
                agent,
                environment_id=lease.environment_id,
                # Checked before every tool call: cancellation and a lost lease both stop new tool execution.
                cancel_check=lambda: beat.lost.is_set() or cancel_requested(sm, lease.experiment_id),
                allowed_tools=spec.tools,
            )
        except Exception as exc:  # noqa: BLE001 - a platform bug must still end as an explicit, retried error
            log.exception("attempt %s crashed in the runner", lease.attempt_id)
            crash = exc
    if crash is not None:  # recorded after the heartbeat stops, so it never reports a spurious lost lease
        _fail(sm, lease, "WORKER_EXCEPTION", f"{type(crash).__name__}: {crash}", True, retry_backoff_seconds)
        return
    duration_ms = (time.monotonic() - started) * 1000
    try:
        finalize_attempt(sm, lease, result, duration_ms=duration_ms, retry_backoff_seconds=retry_backoff_seconds)
    except LeaseLost:
        log.warning("discarding result of attempt %s: lease lost before finalizing", lease.attempt_id)
    except Exception as exc:  # noqa: BLE001
        # The agent ran (and may have spent money): record that explicitly with its usage, never re-run it.
        log.exception("could not persist attempt %s", lease.attempt_id)
        _fail(sm, lease, "PERSIST_FAILED", f"{type(exc).__name__}: {exc}", False, retry_backoff_seconds, result.usage)


def _fail(
    sm: sessionmaker[Session],
    lease: Lease,
    code: str,
    message: str,
    retryable: bool,
    backoff: float,
    usage: UsageSummary | None = None,
) -> None:
    try:
        fail_attempt(
            sm, lease, error_code=code, message=message, retryable=retryable, retry_backoff_seconds=backoff, usage=usage
        )
    except LeaseLost:
        log.warning("could not record failure of attempt %s: lease lost", lease.attempt_id)

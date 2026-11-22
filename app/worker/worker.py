"""The worker loop: bounded concurrency over a PostgreSQL queue, with a periodic expired-lease sweep."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from sqlalchemy import exists, select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.registry import DEFAULT_ADAPTERS, AdapterFactory
from app.db.models import Job
from app.worker.executor import execute
from app.worker.queue import Lease, claim_job, sweep_expired_leases

log = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 2


class Worker:
    def __init__(
        self,
        sm: sessionmaker[Session],
        *,
        worker_id: str | None = None,
        adapters: Mapping[str, AdapterFactory] = DEFAULT_ADAPTERS,
        concurrency: int = DEFAULT_CONCURRENCY,
        lease_seconds: float = 30.0,
        heartbeat_seconds: float = 10.0,
        poll_seconds: float = 1.0,
        sweep_seconds: float = 5.0,
        retry_backoff_seconds: float = 5.0,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if not 0 < heartbeat_seconds < lease_seconds:
            raise ValueError("heartbeat_seconds must be positive and shorter than lease_seconds")
        self.sm = sm
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self.adapters = adapters
        self.concurrency = concurrency
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.poll_seconds = poll_seconds
        self.sweep_seconds = sweep_seconds
        self.retry_backoff_seconds = retry_backoff_seconds

    def _execute(self, lease: Lease) -> None:
        execute(
            self.sm,
            lease,
            adapters=self.adapters,
            lease_seconds=self.lease_seconds,
            heartbeat_seconds=self.heartbeat_seconds,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )

    def _sweep(self) -> list:
        return sweep_expired_leases(self.sm, retry_backoff_seconds=self.retry_backoff_seconds)

    def _claim(self) -> Lease | None:
        return claim_job(self.sm, self.worker_id, lease_seconds=self.lease_seconds)

    def process_one(self) -> bool:
        """Sweep, then claim and run one job in this thread. False if there was nothing to claim."""
        sweep_expired_leases(self.sm, retry_backoff_seconds=self.retry_backoff_seconds)
        lease = self._claim()
        if lease is None:
            return False
        self._execute(lease)
        return True

    def run_forever(self, stop: threading.Event) -> None:
        """Run until ``stop`` is set, then let in-flight attempts finish."""
        self._loop(stop, until_idle=False, timeout=None)

    def run_until_idle(self, timeout: float = 60.0) -> None:
        """Run until no job anywhere is queued or leased (tests, ``make smoke``)."""
        self._loop(threading.Event(), until_idle=True, timeout=timeout)

    def _work_pending(self) -> bool:
        with self.sm() as session:
            return bool(session.scalar(select(exists().where(Job.status.in_(("queued", "leased"))))))

    def _loop(self, stop: threading.Event, *, until_idle: bool, timeout: float | None) -> None:
        deadline = time.monotonic() + timeout if timeout is not None else None
        next_sweep = 0.0
        in_flight: set[Future[None]] = set()
        with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix=f"attempt-{self.worker_id}") as pool:
            while not stop.is_set():
                if time.monotonic() >= next_sweep:
                    self._guard(self._sweep)
                    next_sweep = time.monotonic() + self.sweep_seconds
                while len(in_flight) < self.concurrency:  # bounded: never more than `concurrency` attempts
                    lease = self._guard(self._claim)
                    if lease is None:
                        break
                    in_flight.add(pool.submit(self._execute, lease))
                if until_idle and not in_flight and not self._work_pending():
                    return
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"worker {self.worker_id} did not become idle within {timeout}s")
                if in_flight:
                    done, _ = wait(in_flight, timeout=self.poll_seconds, return_when=FIRST_COMPLETED)
                    for future in done:
                        in_flight.discard(future)
                        if future.exception() is not None:
                            # The lease will expire and the sweeper hands the run on; nothing is lost.
                            log.error("attempt crashed the executor", exc_info=future.exception())
                else:
                    stop.wait(self.poll_seconds)

    def _guard(self, fn, *args):  # type: ignore[no-untyped-def]
        """Database hiccups in the loop are logged and retried on the next tick, never fatal."""
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001
            log.exception("worker %s: %s failed", self.worker_id, getattr(fn, "__name__", fn))
            return None

"""A real worker process for the kill -9 recovery test (not collected by pytest).

Runs scripted agents that take 30 s per attempt, with a 1 s lease renewed every
0.2 s, so killing the process leaves leased jobs whose leases expire quickly.
"""

from __future__ import annotations

import threading

from app.agents.registry import DEFAULT_ADAPTERS
from app.db.session import make_engine, make_sessionmaker
from app.worker.worker import Worker
from tests.postgres.helpers import SlowHonestAgent


def main() -> None:
    adapters = {**DEFAULT_ADAPTERS, "slow_honest": lambda spec: SlowHonestAgent(30.0)}
    worker = Worker(
        make_sessionmaker(make_engine()),
        worker_id="doomed-process",
        adapters=adapters,
        lease_seconds=1.0,
        heartbeat_seconds=0.2,
        poll_seconds=0.05,
    )
    worker.run_forever(threading.Event())


if __name__ == "__main__":
    main()

"""``python -m app.worker``: run a worker until SIGINT/SIGTERM."""

from __future__ import annotations

import argparse
import logging
import signal
import threading

from app.db.session import make_engine, make_sessionmaker
from app.services.catalog import sync_price_versions
from app.worker.worker import DEFAULT_CONCURRENCY, Worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent evaluation worker")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--lease-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    sm = make_sessionmaker(make_engine(pool_size=args.concurrency + 2))
    with sm.begin() as session:
        sync_price_versions(session)
    worker = Worker(
        sm,
        concurrency=args.concurrency,
        lease_seconds=args.lease_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        poll_seconds=args.poll_seconds,
    )
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    logging.getLogger(__name__).info("worker %s started (concurrency %d)", worker.worker_id, worker.concurrency)
    worker.run_forever(stop)


if __name__ == "__main__":
    main()

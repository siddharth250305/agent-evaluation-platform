"""``python -m app.judges --experiment ID``: run the optional communication judge (needs ANTHROPIC_API_KEY)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

from app.accounting import price_for
from app.db.session import make_engine, make_sessionmaker
from app.judges.communication import DEFAULT_JUDGE_MODEL, CommunicationJudge
from app.providers.anthropic_http import AnthropicMessagesProvider
from app.services.judging import judge_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional LLM judge of communication quality (secondary signal)")
    parser.add_argument("--experiment", type=uuid.UUID, required=True)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--limit", type=int, default=None, help="judge at most N attempts (pilot runs)")
    args = parser.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; the judge is optional and the platform works without it.",
              file=sys.stderr)
        return 2
    price = price_for(args.model)
    if price is None:
        print(f"warning: no price version for {args.model}; judge cost will be recorded as unknown", file=sys.stderr)
    judge = CommunicationJudge(AnthropicMessagesProvider(model=args.model, effort=args.effort), price=price)
    counts = judge_experiment(make_sessionmaker(make_engine()), args.experiment, judge, limit=args.limit)
    print(json.dumps({"judge": f"{judge.name}@{judge.version}", "model": args.model, **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

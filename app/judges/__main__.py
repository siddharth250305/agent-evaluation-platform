"""``python -m app.judges --experiment ID``: run the optional communication judge.

Needs the key for the judge model's vendor: ANTHROPIC_API_KEY (default model) or OPENAI_API_KEY (``--model gpt-...``).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Sequence

from app.accounting import price_for
from app.db.session import make_engine, make_sessionmaker
from app.judges.communication import DEFAULT_JUDGE_MODEL, CommunicationJudge
from app.providers import factory
from app.services.judging import judge_experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional LLM judge of communication quality (secondary signal)")
    parser.add_argument("--experiment", type=uuid.UUID, required=True)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--effort", default=None, help="default: low for Claude; none sent for OpenAI")
    parser.add_argument("--limit", type=int, default=None, help="judge at most N attempts (pilot runs)")
    args = parser.parse_args(argv)
    try:
        missing = factory.missing_api_keys([args.model])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if missing:
        print(f"{missing[0]} is not set (needed for {args.model}); the judge is optional and the platform "
              "works without it.", file=sys.stderr)
        return 2
    price = price_for(args.model)
    if price is None:
        print(f"warning: no price version for {args.model}; judge cost will be recorded as unknown", file=sys.stderr)
    effort = args.effort
    if effort is None and factory.provider_for(args.model) == "anthropic":
        effort = "low"  # a rubric verdict does not need deep thinking
    judge = CommunicationJudge(factory.make_provider(args.model, effort=effort), price=price)
    counts = judge_experiment(make_sessionmaker(make_engine()), args.experiment, judge, limit=args.limit)
    print(json.dumps({"judge": f"{judge.name}@{judge.version}", "model": args.model, **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

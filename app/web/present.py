"""Presentation helpers for the dashboard. Pure functions, unit-tested without a database."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

_MISSING = object()


def state_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    """Leaf-level differences between two snapshots; mappings are walked, everything else compared whole."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(set(before) | set(after), key=str):
            child = f"{path}.{key}" if path else str(key)
            changes.extend(state_diff(before.get(key, _MISSING), after.get(key, _MISSING), child))
        return changes
    if before is _MISSING:
        return [{"path": path, "change": "added", "before": None, "after": after}]
    if after is _MISSING:
        return [{"path": path, "change": "removed", "before": before, "after": None}]
    if before != after:
        return [{"path": path, "change": "changed", "before": before, "after": after}]
    return []


def evidence_sequences(evidence: Any) -> set[int]:
    """Trace sequences an evaluator cites: any ``sequence`` or ``*_sequence`` integer, at any depth."""
    found: set[int] = set()
    if isinstance(evidence, dict):
        for key, value in evidence.items():
            if (key == "sequence" or str(key).endswith("_sequence")) and isinstance(value, int) \
                    and not isinstance(value, bool):
                found.add(value)
            else:
                found |= evidence_sequences(value)
    elif isinstance(evidence, list):
        for item in evidence:
            found |= evidence_sequences(item)
    return found


# --- Jinja filters ---------------------------------------------------------------------


def pct(ratio: dict[str, Any] | None) -> str:
    """``62.5% (5/8)``; an empty denominator is ``n/a (0/0)``, never 0%."""
    if not ratio:
        return "n/a"
    value = ratio.get("value")
    shown = "n/a" if value is None else f"{value * 100:.1f}%"
    return f"{shown} ({ratio['numerator']}/{ratio['denominator']})"


def tool_rate(metric: dict[str, Any] | None) -> str:
    """A call-level tool metric: never 0% when nothing applied or when evaluator errors hide the answer."""
    if not metric:
        return "n/a"
    if metric.get("applicability") == "not_applicable":
        return f"not applicable ({metric['numerator']}/{metric['denominator']})"
    if metric.get("applicability") == "unknown":
        return f"unknown ({metric['numerator']}/{metric['denominator']})"
    return pct(metric)


def tool_runs(metric: dict[str, Any] | None) -> str:
    """Which runs contributed, and why the others did not."""
    if not metric:
        return ""
    runs = metric.get("runs", {})
    parts = [f"{runs.get('counted', 0)} runs counted"]
    for key, label in (("not_applicable", "n/a"), ("evaluation_error", "evaluator errors"),
                       ("missing_counts", "without counts"), ("ungraded", "ungraded")):
        if runs.get(key):
            parts.append(f"{runs[key]} {label}")
    return ", ".join(parts)


def pp(delta: float | None) -> str:
    """A rate difference in percentage points."""
    if delta is None:
        return "n/a"
    return f"{delta * 100:+.1f} pp"


def ci(interval: dict[str, Any] | None) -> str:
    if not interval or interval.get("low") is None:
        return "n/a"
    return f"[{interval['low'] * 100:+.1f}, {interval['high'] * 100:+.1f}]"


def ci_rate(interval: dict[str, Any] | None) -> str:
    if not interval or interval.get("low") is None:
        return "n/a"
    return f"[{interval['low'] * 100:.1f}%, {interval['high'] * 100:.1f}%]"


def usd(value: str | Decimal | None) -> str:
    """Money arrives as a decimal string or Decimal; None means unknown, which is never shown as $0."""
    if value is None:
        return "unknown"
    if isinstance(value, Decimal):
        value = format(value.normalize(), "f")
    return f"-${value[1:]}" if value.startswith("-") else f"${value}"


def usd_delta(value: str | None) -> str:
    if value is None:
        return "unknown"
    return usd(value) if value.startswith("-") else f"+${value}"


def ms(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


def ms_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return ("+" if value >= 0 else "-") + ms(abs(value))


def when(value: datetime | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.utcoffset() is None:
        return value.isoformat()
    # PostgreSQL hands timestamptz back in the session time zone; always show UTC.
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def short(value: str | None, n: int = 8) -> str:
    return value[:n] if value else "-"


FILTERS = {
    "pct": pct,
    "tool_rate": tool_rate,
    "tool_runs": tool_runs,
    "pp": pp,
    "ci": ci,
    "ci_rate": ci_rate,
    "usd": usd,
    "usd_delta": usd_delta,
    "ms": ms,
    "ms_delta": ms_delta,
    "when": when,
    "pretty": pretty,
    "short": short,
}

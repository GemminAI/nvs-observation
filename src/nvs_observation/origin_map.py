"""How a capture's value and the Adapter's ``origin_facts()`` become one contract value
(design U6-5 v0.4, 4.0 and 5.1).

Pure rules on plain values; no ``nvs_edge`` and no numpy. A value that is not known is a ``Missing``
(source and reason); the caller turns it into ``nvs_edge``'s ``Unknown2``. The same input gives the
same ``source`` and ``reason`` every time.

``origin_facts()`` writes an unknown value as ``{"source": ..., "reason": ...}`` (U2's
``OriginUnknown.as_dict()``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

#: The reasons ``nvs_edge`` accepts for an unknown value.
UNKNOWN_REASONS: Final = ("unknown", "not_recorded", "unavailable", "failed")
REASON_NOT_RECORDED: Final = "not_recorded"
REASON_UNKNOWN: Final = "unknown"

FACTS_PREFIX: Final = "origin_facts.origin"


@dataclass(frozen=True, slots=True)
class Missing:
    """A value that could not be established: what was looked at, and why it is missing."""

    source: str
    reason: str = REASON_UNKNOWN


@dataclass(frozen=True, slots=True)
class Merged:
    """The value (or ``Missing``) and, when the two sides disagree, the field that disagrees."""

    value: Any
    mismatch: str | None = None


def is_unknown_dict(value: object) -> bool:
    """``{"source": str, "reason": str}`` and nothing else: how ``origin_facts()`` says unknown."""
    return (
        isinstance(value, Mapping)
        and set(value) == {"source", "reason"}
        and isinstance(value["source"], str)
        and isinstance(value["reason"], str)
    )


def missing_from_facts(value: Mapping[str, Any]) -> Missing:
    """U-2: the dictionary's ``source``; its ``reason`` if nvs_edge knows it, else ``unknown``."""
    reason = value["reason"]
    return Missing(value["source"], reason if reason in UNKNOWN_REASONS else REASON_UNKNOWN)


def missing_from_capture(path: str) -> Missing:
    """U-1 (and the capture side of U-3): the capture's path, ``not_recorded``."""
    return Missing(path, REASON_NOT_RECORDED)


def merge_single(
    capture_value: Any, capture_path: str, facts_value: Any, facts_path: str, field: str
) -> Merged:
    """5.1 (a): one value that both the capture and ``origin_facts()`` have.

    ==============  ==================  ===========================================
    capture         origin_facts        result
    ==============  ==================  ===========================================
    A               A                   A
    A               B                   mismatch
    A               unknown             mismatch (a different capture's facts, maybe)
    None            a value             mismatch
    None            unknown             Missing (U-3: the capture's path, not_recorded)
    ==============  ==================  ===========================================

    A value of ``origin_facts()`` that is neither a value nor an unknown dictionary (the key is
    missing, say) is taken as unknown with ``facts_path`` as its source.
    """
    facts_unknown = is_unknown_dict(facts_value) or facts_value is None
    if capture_value is None:
        if facts_unknown:
            return Merged(missing_from_capture(capture_path))
        return Merged(None, field)
    if facts_unknown:
        return Merged(None, field)
    if capture_value != facts_value:
        return Merged(None, field)
    return Merged(capture_value)


def facts_value_or_missing(facts_value: Any, facts_path: str) -> Any:
    """A value only ``origin_facts()`` has: the value, or ``Missing`` (U-2)."""
    if is_unknown_dict(facts_value):
        return missing_from_facts(facts_value)
    if facts_value is None:
        return Missing(facts_path, REASON_UNKNOWN)
    return facts_value


def capture_value_or_missing(capture_value: Any, capture_path: str) -> Any:
    """A value only the capture has: the value, or ``Missing`` (U-1)."""
    return missing_from_capture(capture_path) if capture_value is None else capture_value


def merge_runtime(
    capture_runtime: Mapping[str, Any], facts_runtime: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """5.1 (b): the keys of both ``runtime`` mappings. ``(merged, mismatched fields)``."""
    merged: dict[str, Any] = {}
    mismatches: list[str] = []
    for key in sorted({*capture_runtime, *facts_runtime}, key=str):
        capture_path = f"capture.provenance.runtime.{key}"
        facts_path = f"{FACTS_PREFIX}.runtime.{key}"
        field = f"origin.runtime.{key}"
        if key in capture_runtime and key in facts_runtime:
            result = merge_single(
                capture_runtime[key], capture_path, facts_runtime[key], facts_path, field
            )
            if result.mismatch is not None:
                mismatches.append(result.mismatch)
                continue
            merged[key] = result.value
        elif key in capture_runtime:
            merged[key] = capture_value_or_missing(capture_runtime[key], capture_path)
        else:
            merged[key] = facts_value_or_missing(facts_runtime[key], facts_path)
    return merged, mismatches


def facts_unknowns(origin: Mapping[str, Any], prefix: str = FACTS_PREFIX) -> list[dict[str, str]]:
    """Every unknown dictionary in ``origin_facts()['origin']``, as ``{field, source, reason}``.

    All of them are kept in ``capture_extra`` whichever side was used for the contract (4.0).
    """
    found: list[dict[str, str]] = []
    for key in sorted(origin, key=str):
        value = origin[key]
        path = f"{prefix}.{key}"
        if is_unknown_dict(value):
            found.append({"field": path, "source": value["source"], "reason": value["reason"]})
        elif isinstance(value, Mapping):
            found.extend(facts_unknowns(value, path))
    return found

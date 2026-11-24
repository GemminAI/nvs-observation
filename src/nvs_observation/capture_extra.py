"""``capture_extra``: values of a capture that the contract has no field for (U6-5 v0.4, 5.2).

Nothing is silently turned into a string or dropped. A value is put in as it is, or it is left
out and a line in ``unknowns`` says which path and why. The result has the form::

    {"version": "nvs.capture-extra/1", "values": {...}, "unknowns": [...]}

and it is measured, hashed and written in one canonical JSON (``canonical_json``). That form is
not ``nvs_edge.contract.canonical_bytes`` (which refuses ``float``): ``sampler`` has floats.

This module imports the standard library only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

VERSION: Final = "nvs.capture-extra/1"

REASON_NOT_JSON_VALUE: Final = "not_json_value"
REASON_NOT_JSON_KEY: Final = "not_json_key"
REASON_TOO_DEEP: Final = "too_deep"
REASON_TOO_LARGE: Final = "too_large"

MAX_SAFE_INT: Final = 2**53 - 1
MAX_DEPTH: Final = 16
MAX_VALUE_BYTES: Final = 16 * 1024
MAX_UNKNOWNS: Final = 64
MAX_UNKNOWNS_BYTES: Final = 8 * 1024
MAX_FIELD_CHARS: Final = 256
MAX_DETAIL_CHARS: Final = 128
MAX_TOTAL_BYTES: Final = 64 * 1024

#: Small values that an audit needs. They are never dropped to make room (5.2, step 4).
KEEP_KEYS: Final = frozenset(
    {"max_tokens", "raw_encoding", "recorded_at", "adapter_notes", "origin_facts_unknowns"}
)

_ABSENT: Final = object()
#: What ``clean_value`` returns when nothing of the value can be kept.
ABSENT: Final = _ABSENT


def canonical_json(value: Any) -> bytes:
    """The one JSON of this module: sorted keys, no spaces, UTF-8 as it is, no NaN or Inf.

    Python's ``json`` escapes only ``"``, ``\\`` and U+0000..U+001F here (``\\b \\f \\n \\r \\t`` in
    the short form, the rest as lowercase ``\\u00xx``); keys are ordered by code point; a float is
    written by ``repr``. No normalization of the text is done. A lone surrogate raises
    ``UnicodeEncodeError`` -- values are cleaned by ``clean_value`` first.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utf8_ok(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _type_name(value: object) -> str:
    return type(value).__name__[:MAX_DETAIL_CHARS]


def _field(path: str) -> str:
    """A path of at most 256 characters, or ``sha256:`` of its UTF-8 bytes (a stable stand-in)."""
    if len(path) <= MAX_FIELD_CHARS:
        return path
    return "sha256:" + sha256_hex(path.encode("utf-8", "surrogatepass"))


def _unknown(path: str, reason: str, detail: str) -> dict[str, str]:
    return {"field": _field(path), "reason": reason, "detail": detail[:MAX_DETAIL_CHARS]}


def _walk(value: Any, path: str, depth: int, unknowns: list[dict[str, Any]]) -> Any:
    """The JSON form of ``value``, or ``_ABSENT`` (a line is then added to ``unknowns``)."""
    kind = type(value)
    if kind is bool or value is None:
        return value
    if kind is int:
        if -MAX_SAFE_INT <= value <= MAX_SAFE_INT:
            return value
        unknowns.append(_unknown(path, REASON_NOT_JSON_VALUE, "int out of range"))
        return _ABSENT
    if kind is float:
        if math.isfinite(value):
            return value
        unknowns.append(_unknown(path, REASON_NOT_JSON_VALUE, "float not finite"))
        return _ABSENT
    if kind is str:
        if _utf8_ok(value):
            return value
        unknowns.append(_unknown(path, REASON_NOT_JSON_VALUE, "str not UTF-8"))
        return _ABSENT
    if isinstance(value, Mapping | list | tuple):
        if depth > MAX_DEPTH:
            unknowns.append(_unknown(path, REASON_TOO_DEEP, _type_name(value)))
            return _ABSENT
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str or not _utf8_ok(key):
                    unknowns.append(_unknown(path, REASON_NOT_JSON_KEY, _type_name(key)))
                    continue
                cleaned = _walk(item, f"{path}.{key}", depth + 1, unknowns)
                if cleaned is not _ABSENT:
                    out[key] = cleaned
            return out
        items: list[Any] = []
        for index, item in enumerate(value):
            cleaned = _walk(item, f"{path}[{index}]", depth + 1, unknowns)
            if cleaned is not _ABSENT:
                items.append(cleaned)
        return items
    unknowns.append(_unknown(path, REASON_NOT_JSON_VALUE, _type_name(value)))
    return _ABSENT


def clean_value(value: Any, path: str) -> tuple[Any, list[dict[str, Any]]]:
    """``(json_form, unknowns)``; ``json_form`` is ``ABSENT`` if nothing of ``value`` can be kept.

    ``tuple`` becomes ``list`` and a ``Mapping`` becomes ``dict``; nothing else is converted. A bad
    element of a list is left out and its index is the one of the original list (``a[3]``).
    """
    unknowns: list[dict[str, Any]] = []
    cleaned = _walk(value, path, 1, unknowns)
    return cleaned, unknowns


def _too_large(field: str, canonical: bytes) -> dict[str, Any]:
    return {
        "field": _field(field),
        "reason": REASON_TOO_LARGE,
        "detail": f"{len(canonical)} bytes",
        "sha256": sha256_hex(canonical),
    }


def _order(unknowns: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        unknowns,
        key=lambda u: (u["field"], u["reason"], u.get("detail", ""), u.get("sha256", "")),
    )


@dataclass(frozen=True, slots=True)
class ExtraResult:
    """``extra`` is the ``capture_extra`` document, or ``None`` when it overflows.

    ``overflow`` (only then) holds the sizes that the refusal ``capture_extra_overflow`` reports:
    ``total_bytes``, ``unknowns_count``, ``unknowns_bytes`` and ``reasons``.
    """

    extra: dict[str, Any] | None
    overflow: dict[str, Any] | None

    @property
    def overflowed(self) -> bool:
        return self.extra is None


def build_capture_extra(
    values: Mapping[str, Any],
    *,
    extra_unknowns: Iterable[Mapping[str, Any]] = (),
    keep_keys: frozenset[str] = KEEP_KEYS,
) -> ExtraResult:
    """Build ``capture_extra`` from ``values`` (key -> any value) by the rules of 5.2.

    ``extra_unknowns`` are lines the caller already has (``field``, ``reason``, ``detail``); they
    are measured with the others. The same input always gives the same bytes.
    """
    cleaned: dict[str, Any] = {}
    unknowns: list[dict[str, Any]] = [dict(u) for u in extra_unknowns]
    for key in sorted(values, key=lambda k: (type(k).__name__, str(k))):
        if type(key) is not str or not _utf8_ok(key):
            unknowns.append(_unknown("values", REASON_NOT_JSON_KEY, _type_name(key)))
            continue
        form, found = clean_value(values[key], key)
        unknowns.extend(found)
        if form is not _ABSENT:
            cleaned[key] = form

    reasons: list[str] = []
    sizes: dict[str, int] = {}
    for key in sorted(cleaned):
        encoded = canonical_json(cleaned[key])
        sizes[key] = len(encoded)
        if len(encoded) > MAX_VALUE_BYTES:
            unknowns.append(_too_large(key, encoded))
            del cleaned[key]
            if key in keep_keys:
                reasons.append(f"kept key too large: {key}")

    def document() -> dict[str, Any]:
        return {"version": VERSION, "values": cleaned, "unknowns": _order(unknowns)}

    total = len(canonical_json(document()))
    if total > MAX_TOTAL_BYTES:
        droppable = sorted(
            (k for k in cleaned if k not in keep_keys), key=lambda k: (-sizes[k], k)
        )
        for key in droppable:
            unknowns.append(_too_large(key, canonical_json(cleaned[key])))
            del cleaned[key]
            total = len(canonical_json(document()))
            if total <= MAX_TOTAL_BYTES:
                break
        if total > MAX_TOTAL_BYTES:
            reasons.append("total too large after dropping every droppable key")

    result = document()
    unknowns_bytes = len(canonical_json(result["unknowns"]))
    if len(result["unknowns"]) > MAX_UNKNOWNS:
        reasons.append("too many unknowns")
    if unknowns_bytes > MAX_UNKNOWNS_BYTES:
        reasons.append("unknowns too large")
    if reasons:
        return ExtraResult(
            extra=None,
            overflow={
                "total_bytes": total,
                "unknowns_count": len(result["unknowns"]),
                "unknowns_bytes": unknowns_bytes,
                "reasons": reasons,
            },
        )
    return ExtraResult(extra=result, overflow=None)

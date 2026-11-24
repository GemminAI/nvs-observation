"""``contract_from_capture``: a contract v2 document, its RAW files and instrument records, built
from one capture and its ``origin_facts()`` (design U6-5 v0.4, 6.1, 7).

The mapping and the checks are ``capture_map``'s. This module adds what needs the RAW:

1. ``encode_raw(states, created_at_ms=0, dimension=D)`` gives the file and the ``trajectory_id``
   (the SHA-256 of the rows' bytes, not of the file). ``created_at_ms`` is the fixed sentinel ``0``:
   it is not a time and must not be read as 1970-01-01. The time of the record is
   ``capture_extra.values.recorded_at``.
2. ``decode_raw`` reads the file back and the rows must be bit for bit the ones that went in.
3. ``instrument_record(rows read back, trajectory_id)``: the second argument is the
   ``trajectory_id``, which is what the verification (G2) compares with the file's header.

When anything is refused, no contract, RAW or instrument record is returned (never a part of
them); a record of the capture is always possible: ``capture_block`` and ``capture_extra`` are
returned for it. They are not given to ``build_record`` here (that is U6-5a' and U6-5d).

This module needs numpy and ``nvs_edge`` (through ``capture_edge``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from nvs_observation import capture_edge
from nvs_observation.adapter_session import CapturedWithOrigin
from nvs_observation.capture_check import np
from nvs_observation.capture_extra import build_capture_extra
from nvs_observation.capture_map import (
    CAPTURE_EXTRA_OVERFLOW,
    CONTRACT_NOT_BUILDABLE,
    Mapped,
    describe_capture,
    map_capture,
)

#: ``created_at_ms`` of every RAW file made here: a sentinel, not a time (7).
CREATED_AT_SENTINEL: Final = 0
RAW_ENCODING: Final = {
    "created_at_ms": CREATED_AT_SENTINEL,
    "created_at_ms_meaning": "fixed_sentinel_not_a_time",
}


@dataclass(frozen=True, slots=True)
class ContractBuild:
    """What ``contract_from_capture`` returns.

    ``contract`` is ``None`` when ``refusals`` is not empty, and then ``raw_bytes`` and
    ``instrument_records`` are empty. ``refusal_details`` says, for each refusal, which check or
    field. ``capture_block`` is the ``capture`` of a record (``record.capture_block``).
    ``capture_extra`` is the ``capture_extra`` of a record (5.2), or ``None`` after
    ``capture_extra_overflow``.
    """

    contract: dict[str, Any] | None
    raw_bytes: dict[str, bytes]
    instrument_records: dict[str, dict[str, Any]]
    refusals: tuple[str, ...]
    refusal_details: tuple[str, ...]
    capture_block: dict[str, Any]
    capture_extra: dict[str, Any] | None
    session_id: str = field(default="")


def _now() -> datetime:
    return datetime.now().astimezone()


def _refused(
    mapped: Mapped,
    capture: Any,
    refusals: tuple[str, ...],
    details: tuple[str, ...],
    extra: dict[str, Any] | None,
    session_id: str,
) -> ContractBuild:
    return ContractBuild(
        contract=None,
        raw_bytes={},
        instrument_records={},
        refusals=refusals,
        refusal_details=details,
        capture_block=describe_capture(capture, mapped.check),
        capture_extra=extra,
        session_id=session_id,
    )


def contract_from_capture(
    capture: Any,
    origin_facts: Mapping[str, Any],
    *,
    session_id: str,
    synthetic: bool,
    last_capture_check: str | None = None,
    clock: Callable[[], datetime] = _now,
) -> ContractBuild:
    """Build the contract of ``capture``, or refuse it. Does not raise for a bad capture.

    ``capture`` and ``origin_facts`` must be one pair made by the rule of 3.1 (L-a): taken from the
    same Adapter instance, one right after the other, under one lock -- ``AdapterSession`` does
    this. Values that agree do not show that the facts were made from this capture. Nothing here
    can check it, except that ``last_capture_check`` (``AdapterSession``'s) is ``not_last``.

    ``session_id`` is what the contract records; ``synthetic`` has no default. ``clock`` gives
    the ``recorded_at`` of the record (a time with its offset); a test passes a fixed one.
    ``capture`` and ``origin_facts`` are not changed.
    """
    mapped = map_capture(
        capture,
        origin_facts,
        session_id=session_id,
        synthetic=synthetic,
        last_capture_check=last_capture_check,
    )
    refusals = list(mapped.refusals)
    details = list(mapped.details)
    raw_bytes: dict[str, bytes] = {}
    instrument_records: dict[str, dict[str, Any]] = {}
    trajectory_ids: dict[str, str] = {}
    edge, _ = capture_edge.current()

    if not refusals and edge is not None:
        try:
            for point in mapped.points:
                states = capture.states[point]
                data, header = edge.encode_raw(
                    states, created_at_ms=CREATED_AT_SENTINEL, dimension=capture.dimension
                )
                decoded, back = edge.decode_raw(data, dimension=None)
                expected = np.ascontiguousarray(states, dtype="<f4").tobytes()
                if back.trajectory_id != header.trajectory_id or (
                    np.ascontiguousarray(decoded, dtype="<f4").tobytes() != expected
                ):
                    raise ValueError(f"the RAW file of {point} does not read back as its rows")
                raw_bytes[point] = data
                trajectory_ids[point] = header.trajectory_id
                instrument_records[point] = edge.instrument_record(decoded, header.trajectory_id)
        except Exception as error:  # noqa: BLE001 -- RawStoreError, ValueError, a zstd error
            refusals.append(CONTRACT_NOT_BUILDABLE)
            details.append(f"{CONTRACT_NOT_BUILDABLE}: RAW {type(error).__name__}: {error}")
    contract: dict[str, Any] | None = None
    if not refusals and edge is not None:
        try:
            contract = edge.build_contract_v2(**mapped.contract_parts(trajectory_ids))
        except edge.contract_error as error:
            refusals.append(CONTRACT_NOT_BUILDABLE)
            details.append(f"{CONTRACT_NOT_BUILDABLE}: {error}")

    values = dict(mapped.extra_values)
    values["recorded_at"] = clock().isoformat(timespec="seconds")
    if contract is not None:
        values["raw_encoding"] = dict(RAW_ENCODING)
    elif "origin" not in values:
        # refused after the mapping (the RAW or the contract): the origin is kept as for any refusal
        facts_origin = origin_facts.get("origin") if isinstance(origin_facts, Mapping) else None
        if facts_origin is not None:
            values["origin"] = facts_origin
    result = build_capture_extra(values, extra_unknowns=mapped.extra_unknowns)
    if result.overflowed:
        overflow = result.overflow or {}
        refusals.append(CAPTURE_EXTRA_OVERFLOW)
        details.append(
            f"{CAPTURE_EXTRA_OVERFLOW}: total {overflow.get('total_bytes')} bytes, "
            f"{overflow.get('unknowns_count')} unknowns of {overflow.get('unknowns_bytes')} bytes; "
            + "; ".join(overflow.get("reasons", ()))
        )
        contract = None
    if refusals:
        return _refused(
            mapped,
            capture,
            tuple(dict.fromkeys(refusals)),
            tuple(details),
            result.extra,
            session_id,
        )
    return ContractBuild(
        contract=contract,
        raw_bytes=raw_bytes,
        instrument_records=instrument_records,
        refusals=(),
        refusal_details=(),
        capture_block=describe_capture(capture, mapped.check),
        capture_extra=result.extra,
        session_id=session_id,
    )


def contract_from_captured(
    captured: CapturedWithOrigin,
    *,
    session_id: str,
    synthetic: bool,
    clock: Callable[[], datetime] = _now,
) -> ContractBuild:
    """``contract_from_capture`` for a pair made by ``AdapterSession`` (L-c, L-e)."""
    return contract_from_capture(
        captured.capture,
        captured.origin_facts,
        session_id=session_id,
        synthetic=synthetic,
        last_capture_check=captured.last_capture_check,
        clock=clock,
    )

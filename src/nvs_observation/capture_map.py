"""A capture and its ``origin_facts()`` mapped to the fields of contract v2 (U6-5 v0.4, 4, 5.1).

``map_capture`` does the part that needs no RAW: the checks of 5.3, the comparison of the two
sources of provenance (5.1), the space materials (4.3) and the fields of ``binding``, ``raw``,
``positions`` and ``origin`` (4.1-4.5). Every value of the capture goes to one of: a field of the
contract with the same value, an ``Unknown2`` with its source and reason, ``capture_extra``, or a
row of the table in the design that says why it is not carried. Nothing is dropped silently.

Refusals are returned, not raised. ``trajectory_id`` of each position is known only after the RAW
is encoded (U6-5c), so ``Mapped.contract_parts`` takes it as an argument.

This module needs numpy and ``nvs_edge`` (through ``capture_edge``). It does not import
``sensos_inference``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from nvs_observation import capture_edge, origin_map
from nvs_observation.capture_check import CheckResult, check_capture
from nvs_observation.capture_edge import CaptureEdge
from nvs_observation.contracts.capture_record import (
    REASON_CAPTURE_EXTRA_OVERFLOW,
    REASON_CAPTURE_INVALID,
    REASON_CONTRACT_NOT_BUILDABLE,
    REASON_EMPTY_GENERATION,
    REASON_ORIGIN_FACTS_INVALID,
    REASON_ORIGIN_MISMATCH,
    REASON_SPACE_COLLISION,
    capture_block,
)
from nvs_observation.origin_map import Missing

ALIGNMENT_DECIDES_NEXT: Final = "decides-next/1"
TOKEN_SOURCE: Final = "model-input-ids"
#: ``raw.excluded`` is always these four words, in this order (4.5).
EXCLUDED: Final = ("other-layers", "attention", "logits", "kv")
_CAST_EXACT_DTYPES: Final = frozenset({"float16", "bfloat16", "float32"})

EMPTY_GENERATION: Final = REASON_EMPTY_GENERATION
CAPTURE_INVALID: Final = REASON_CAPTURE_INVALID
ORIGIN_MISMATCH: Final = REASON_ORIGIN_MISMATCH
ORIGIN_FACTS_INVALID: Final = REASON_ORIGIN_FACTS_INVALID
SPACE_COLLISION: Final = REASON_SPACE_COLLISION
CONTRACT_NOT_BUILDABLE: Final = REASON_CONTRACT_NOT_BUILDABLE
CAPTURE_EXTRA_OVERFLOW: Final = REASON_CAPTURE_EXTRA_OVERFLOW
CAPTURE_NOT_LAST: Final = "capture_not_last"

_ORIGIN_FACT_NOTES: Final = ("capabilities_phase", "observed_state_dtype", "version_policy")
_UNSET: Final = object()


@dataclass(frozen=True, slots=True)
class Mapped:
    """What ``map_capture`` found. ``parts`` is ``None`` when anything is refused.

    ``extra_values`` and ``extra_unknowns`` are the input of ``capture_extra`` (U6-5c adds
    ``raw_encoding`` and ``recorded_at`` and builds it).
    """

    refusals: tuple[str, ...]
    details: tuple[str, ...]
    check: CheckResult
    parts: Mapping[str, Any] | None
    extra_values: dict[str, Any]
    extra_unknowns: tuple[dict[str, str], ...]
    points: tuple[str, ...] = ()
    session_id: str = ""
    _edge: CaptureEdge | None = field(default=None, repr=False)

    def contract_parts(self, trajectory_ids: Mapping[str, str]) -> dict[str, Any]:
        """The arguments of ``build_contract_v2`` (``Missing`` turned into ``Unknown2``)."""
        if self.parts is None or self._edge is None:
            raise ValueError("nothing was mapped: the capture was refused")
        edge = self._edge
        positions = []
        for position in self.parts["positions"]:
            item = dict(position)
            item["trajectory_id"] = trajectory_ids[item["point"]]
            positions.append(item)
        return {
            "session_id": self.session_id,
            "binding": _unknown2(self.parts["binding"], edge),
            "raw": _unknown2(self.parts["raw"], edge),
            "positions": _unknown2(positions, edge),
            "origin": _unknown2(self.parts["origin"], edge),
        }


def _unknown2(value: Any, edge: CaptureEdge) -> Any:
    if isinstance(value, Missing):
        return edge.unknown2(value.source, value.reason)
    if isinstance(value, Mapping):
        return {key: _unknown2(item, edge) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_unknown2(item, edge) for item in value]
    return value


def space_materials(capture: Any, point: str, edge: CaptureEdge) -> dict[str, Any]:
    """The six materials of one point's space (4.3); ``layer`` keeps its type."""
    return {
        "model_build_id": capture.model_build_id,
        "layer": edge.layer_label(point),
        "row_rule": capture.row_rule[point],
        "compute_environment": capture.compute_environment,
        "dimension": capture.dimension,
        "space_rule": edge.space_rule,
    }


def space_collisions(capture: Any, edge: CaptureEdge) -> list[tuple[str, str]]:
    """Pairs of points with the same ``(type(layer), layer, row_rule)`` (as U1 does)."""
    seen: dict[tuple[str, Any, str], str] = {}
    out: list[tuple[str, str]] = []
    for point in capture.points:
        layer = edge.layer_label(point)
        key = (type(layer).__name__, layer, capture.row_rule[point])
        if key in seen:
            out.append((seen[key], point))
        else:
            seen[key] = point
    return out


def describe_capture(capture: Any, check: CheckResult | None = None) -> dict[str, Any]:
    """The ``capture`` part of a record, as far as the capture can be read (else ``None``)."""
    provenance = getattr(capture, "provenance", None)

    def length(name: str, ids: Any) -> Any:
        if ids is not None:
            return int(len(ids))
        try:
            return len(getattr(capture, name))
        except TypeError:
            return None

    points = getattr(capture, "points", ())
    return capture_block(
        n_in=length("prompt_ids", None if check is None else check.prompt_ids),
        n_gen=length("token_ids", None if check is None else check.token_ids),
        finish_reason=getattr(capture, "finish_reason", None),
        points=[p for p in points if isinstance(p, str)]
        if isinstance(points, (tuple, list))
        else [],
        model_build_id=getattr(capture, "model_build_id", None),
        compute_environment=getattr(capture, "compute_environment", None),
        adapter=getattr(capture, "adapter", None),
        prompt_ids_sha256=getattr(provenance, "prompt_ids_sha256", None),
        gen_token_ids_sha256=getattr(provenance, "gen_token_ids_sha256", None),
    )


def _plain(value: Any) -> Any:
    """A copy of a Mapping/tuple as dict/list, for ``capture_extra`` (which cleans it again)."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _extra_values(
    capture: Any, origin_facts: Any, facts_origin: Mapping[str, Any], last_capture_check: str | None
) -> dict[str, Any]:
    provenance = getattr(capture, "provenance", None)
    values: dict[str, Any] = {}
    for key, source in (
        ("sampler", "sampler"),
        ("max_tokens", "max_tokens"),
        ("chat_template_kwargs", "chat_template_kwargs"),
    ):
        found = getattr(provenance, source, _UNSET)
        if found is not _UNSET:
            values[key] = _plain(found)
    unknowns = getattr(capture, "unknowns", _UNSET)
    if unknowns is not _UNSET:
        values["capture_unknowns"] = _plain(unknowns)
    notes: dict[str, Any] = {}
    if isinstance(origin_facts, Mapping):
        for name in _ORIGIN_FACT_NOTES:
            if name in origin_facts:
                notes[name] = _plain(origin_facts[name])
    if last_capture_check is not None:
        notes["last_capture_check"] = last_capture_check
    values["adapter_notes"] = notes
    values["origin_facts_unknowns"] = origin_map.facts_unknowns(facts_origin)
    return values


def map_capture(
    capture: Any,
    origin_facts: Mapping[str, Any],
    *,
    session_id: str,
    synthetic: bool,
    last_capture_check: str | None = None,
) -> Mapped:
    """Check ``capture`` and map it with ``origin_facts`` to the fields of contract v2.

    ``capture`` and ``origin_facts`` must be one pair made by the rule of 3.1 (``AdapterSession``).
    Neither is changed. Never raises for a bad capture: see ``Mapped.refusals``.
    """
    refusals: list[tuple[str, str]] = []

    def refuse(code: str, detail: str) -> None:
        refusals.append((code, detail))

    check = check_capture(capture)
    for problem in check.problems:
        refuse(problem.code, problem.detail)
    facts_origin_raw = origin_facts.get("origin") if isinstance(origin_facts, Mapping) else None
    if not isinstance(facts_origin_raw, Mapping):
        facts_origin: Mapping[str, Any] = {}
        refuse(ORIGIN_FACTS_INVALID, "origin_facts.origin is missing or not a mapping")
    else:
        facts_origin = facts_origin_raw
    edge, failure = capture_edge.current()
    if edge is None and failure is not None:
        refuse(failure.code, failure.detail)
    if last_capture_check == "not_last":
        refuse(ORIGIN_MISMATCH, CAPTURE_NOT_LAST)
    if check.empty:
        refuse(EMPTY_GENERATION, "n_gen is 0")

    extra_values = _extra_values(capture, origin_facts, facts_origin, last_capture_check)
    extra_unknowns: list[dict[str, str]] = []

    def finish(parts: Mapping[str, Any] | None) -> Mapped:
        if refusals:
            if facts_origin_raw is not None:
                extra_values["origin"] = _plain(facts_origin_raw)
        codes = tuple(dict.fromkeys(code for code, _ in refusals))
        points = getattr(capture, "points", ())
        return Mapped(
            refusals=codes,
            details=tuple(f"{code}: {detail}" for code, detail in refusals),
            check=check,
            parts=None if refusals else parts,
            extra_values=extra_values,
            extra_unknowns=tuple(extra_unknowns),
            points=tuple(points) if isinstance(points, (tuple, list)) else (),
            session_id=session_id,
            _edge=edge,
        )

    if check.problems or check.empty or edge is None:
        return finish(None)

    provenance = capture.provenance
    points: tuple[str, ...] = tuple(capture.points)
    mismatches: list[str] = []

    # -- 4.3: the points, and the collisions of their spaces ------------------------------------
    for point in points:
        if edge.point_pattern.fullmatch(point) is None:
            refuse(CAPTURE_INVALID, f"V-1 point {point!r} does not fit the point names")
    if refusals:
        return finish(None)
    for first, second in space_collisions(capture, edge):
        refuse(SPACE_COLLISION, f"{first} and {second} have the same space")

    # -- 5.1: the two sources of provenance -------------------------------------------------------
    def compare(field_name: str, capture_value: Any, capture_path: str, facts_key: str) -> Any:
        merged = origin_map.merge_single(
            capture_value,
            capture_path,
            facts_origin.get(facts_key),
            f"{origin_map.FACTS_PREFIX}.{facts_key}",
            field_name,
        )
        if merged.mismatch is not None:
            mismatches.append(merged.mismatch)
        return merged.value

    model = compare("origin.model", provenance.model_ref, "capture.provenance.model_ref", "model")
    revision = compare(
        "origin.revision", provenance.revision, "capture.provenance.revision", "revision"
    )
    weights = compare(
        "origin.weights_sha256",
        provenance.weights_sha256,
        "capture.provenance.weights_sha256",
        "weights_sha256",
    )
    capture_runtime = provenance.runtime if isinstance(provenance.runtime, Mapping) else {}
    facts_runtime_raw = facts_origin.get("runtime")
    facts_runtime = facts_runtime_raw if isinstance(facts_runtime_raw, Mapping) else {}
    runtime, runtime_mismatches = origin_map.merge_runtime(capture_runtime, facts_runtime)
    mismatches.extend(runtime_mismatches)
    backend_field = "origin.runtime.backend"
    if backend_field not in runtime_mismatches and (
        "backend" not in runtime or isinstance(runtime["backend"], Missing)
    ):
        refuse(CAPTURE_INVALID, "V-1 origin.runtime.backend is unknown")

    facts_capture_raw = facts_origin.get("capture")
    facts_capture = facts_capture_raw if isinstance(facts_capture_raw, Mapping) else {}
    adapter = origin_map.merge_single(
        capture.adapter,
        "capture.adapter",
        facts_capture.get("adapter"),
        f"{origin_map.FACTS_PREFIX}.capture.adapter",
        "origin.capture.adapter",
    )
    if adapter.mismatch is not None:
        mismatches.append(adapter.mismatch)
    if facts_capture.get("alignment") != ALIGNMENT_DECIDES_NEXT:
        mismatches.append("origin.capture.alignment")
    capture_origin: dict[str, Any] = {
        "package": origin_map.facts_value_or_missing(
            facts_capture.get("package"), f"{origin_map.FACTS_PREFIX}.capture.package"
        ),
        "version": origin_map.facts_value_or_missing(
            facts_capture.get("version"), f"{origin_map.FACTS_PREFIX}.capture.version"
        ),
        "adapter": adapter.value,
        "alignment": ALIGNMENT_DECIDES_NEXT,
    }
    commit = facts_capture.get("source_commit", _UNSET)
    if commit is not _UNSET:
        if isinstance(commit, str) and commit:
            capture_origin["source_commit"] = commit
        else:
            extra_values["adapter_notes"]["source_commit_not_used"] = "not a non-empty str"
    for field_name in mismatches:
        refuse(ORIGIN_MISMATCH, field_name)
    if refusals:
        return finish(None)

    # -- 4.1: binding -----------------------------------------------------------------------------
    special = provenance.special_tokens_in
    if special is None:
        special_value: Any = origin_map.missing_from_capture("capture.provenance.special_tokens_in")
    else:
        try:
            special_value = [[int(position), int(token)] for position, token in special]
        except (TypeError, ValueError):
            refuse(CAPTURE_INVALID, "V-1 special_tokens_in is not a sequence of (position, id)")
            return finish(None)
    eos_ids = provenance.eos_ids
    binding: dict[str, Any] = {
        "alignment": ALIGNMENT_DECIDES_NEXT,
        "token_source": TOKEN_SOURCE,
        "n_in": check.n_in,
        "n_gen": check.n_gen,
        "special_tokens_in": special_value,
        "tokenizer_sha256": origin_map.capture_value_or_missing(
            provenance.tokenizer_sha256, "capture.provenance.tokenizer_sha256"
        ),
        "chat_template_sha256": origin_map.capture_value_or_missing(
            provenance.chat_template_sha256, "capture.provenance.chat_template_sha256"
        ),
        "prompt_ids_sha256": provenance.prompt_ids_sha256,
        "gen_token_ids_sha256": provenance.gen_token_ids_sha256,
        "finish_reason": capture.finish_reason,
        "eos_ids": (
            origin_map.missing_from_capture("capture.provenance.eos_ids")
            if eos_ids is None
            else [int(i) for i in eos_ids]
        ),
    }

    # -- 4.2: raw ---------------------------------------------------------------------------------
    state_dtype = capture.state_dtype
    dtype_known = isinstance(state_dtype, str) and state_dtype != "unknown"
    raw: dict[str, Any] = {
        "state_dtype": state_dtype
        if dtype_known
        else origin_map.missing_from_capture("capture.state_dtype"),
        "stored_dtype": "float32",
        "cast_exact": True
        if dtype_known and state_dtype in _CAST_EXACT_DTYPES
        else origin_map.missing_from_capture("capture.state_dtype"),
        "excluded": list(EXCLUDED),
    }

    # -- 4.3: positions (trajectory_id is added by contract_parts) ------------------------
    n_in, n_gen = check.n_in, check.n_gen
    positions: list[dict[str, Any]] = []
    for point in points:
        materials = space_materials(capture, point, edge)
        positions.append(
            {
                "point": point,
                "row_rule": capture.row_rule[point],
                "alignment": ALIGNMENT_DECIDES_NEXT,
                "dimension": capture.dimension,
                "final_norm_applied": capture.final_norm_applied[point],
                "stored_range": {"steps": n_gen, "first_row_pos": n_in - 1},
                "space": materials,
                "k_path": None,
            }
        )

    # -- 4.4: origin ------------------------------------------------------------------------------
    try:
        edge_version: Any = edge.source_build_id()
    except Exception as error:  # noqa: BLE001 -- OSError for an unreadable source, anything else
        edge_version = Missing("nvs_edge.client.source_build_id", "failed")
        extra_unknowns.append(
            {"field": "origin.edge_version", "reason": "failed", "detail": type(error).__name__}
        )
    origin: dict[str, Any] = {
        "model": model,
        "model_build_id": capture.model_build_id,
        "revision": revision,
        "weights_sha256": weights,
        "quantization": origin_map.facts_value_or_missing(
            facts_origin.get("quantization"), f"{origin_map.FACTS_PREFIX}.quantization"
        ),
        "runtime": runtime,
        "hardware": origin_map.facts_value_or_missing(
            facts_origin.get("hardware"), f"{origin_map.FACTS_PREFIX}.hardware"
        ),
        "blas": Missing("capture", origin_map.REASON_NOT_RECORDED),
        "compute_environment": capture.compute_environment,
        "capture": capture_origin,
        "instruments": [
            {
                "name": edge.instrument_name,
                "version": edge.instrument_version,
                "settings_sha256": edge.instrument_settings_sha256(),
            }
        ],
        "edge_version": edge_version,
        "synthetic": synthetic,
    }
    return finish(
        copy.deepcopy({"binding": binding, "raw": raw, "positions": positions, "origin": origin})
    )


def points_of(mapped: Mapped) -> Sequence[str]:
    return mapped.points

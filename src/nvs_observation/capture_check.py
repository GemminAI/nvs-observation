"""The checks U6-5 makes on a capture it is given, V-0 .. V-7 (design U6-5 v0.4, 5.3).

U6-5 does not import U1, so it cannot assume that a capture passed U1's ``validate_capture``. The
checks are made in order; a failure of V-0 stops them (the later ones need ``n_in`` and ``n_gen``).
Arrays are only read, never changed.

This module needs numpy (loaded with ``importlib``, see below). It does not import
``sensos_inference``.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from nvs_observation.contracts.capture_record import (
    REASON_CAPTURE_INVALID,
    REASON_DIMENSION_OUT_OF_RANGE,
    REASON_STEPS_OUT_OF_RANGE,
)
from nvs_observation.contracts.constants import MAX_DIMENSION

# numpy is loaded by name (as ``contract._load_edge`` loads nvs_edge): the existing test T-U6-E09
# allows no ``import numpy`` statement in any module of this package.
np: Any = importlib.import_module("numpy")

CAPTURE_INVALID: Final = REASON_CAPTURE_INVALID
STEPS_OUT_OF_RANGE: Final = REASON_STEPS_OUT_OF_RANGE
DIMENSION_OUT_OF_RANGE: Final = REASON_DIMENSION_OUT_OF_RANGE

MAX_STEPS: Final = 65535
_INT64_MAX: Final = 2**63 - 1

CAPTURE_ATTRIBUTES: Final = (
    "adapter",
    "model_build_id",
    "compute_environment",
    "dimension",
    "prompt_ids",
    "token_ids",
    "finish_reason",
    "points",
    "states",
    "row_rule",
    "row_pos",
    "row_decides",
    "final_norm_applied",
    "state_dtype",
    "provenance",
    "unknowns",
)
PROVENANCE_ATTRIBUTES: Final = (
    "model_ref",
    "revision",
    "weights_sha256",
    "tokenizer_sha256",
    "chat_template_sha256",
    "special_tokens_in",
    "prompt_ids_sha256",
    "gen_token_ids_sha256",
    "runtime",
    "sampler",
    "max_tokens",
    "chat_template_kwargs",
    "eos_ids",
)

_MISSING: Final = object()


@dataclass(frozen=True, slots=True)
class Problem:
    """``code`` is a refusal reason; ``detail`` names the check (``V-3``) and what failed."""

    code: str
    detail: str


@dataclass(slots=True)
class CheckResult:
    """The ids are int64 arrays once V-0 passed. ``empty`` is ``n_gen == 0``."""

    problems: list[Problem] = field(default_factory=list)
    prompt_ids: Any = None
    token_ids: Any = None
    empty: bool = False

    @property
    def n_in(self) -> int:
        return 0 if self.prompt_ids is None else int(len(self.prompt_ids))

    @property
    def n_gen(self) -> int:
        return 0 if self.token_ids is None else int(len(self.token_ids))

    @property
    def usable(self) -> bool:
        """V-0 passed: ``prompt_ids`` and ``token_ids`` are arrays of ids."""
        return self.prompt_ids is not None and self.token_ids is not None


def ids_sha256(ids: Any) -> str:
    """SHA-256 of the ids as little-endian int64 (U1's ``_ids_sha256``)."""
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _as_ids(values: Any) -> tuple[Any, str | None]:
    """``(int64 array, None)`` for a 1-D sequence of non-negative ints, else ``(None, why)``."""
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            return None, f"ndim is {values.ndim}"
        if values.dtype.kind not in "iu":
            return None, f"dtype is {values.dtype}"
        if values.size and values.dtype.kind == "u" and int(values.max()) > _INT64_MAX:
            return None, "an id does not fit in int64"
        if values.size and int(values.min()) < 0:
            return None, "an id is negative"
        return values.astype(np.int64), None
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        return None, f"{type(values).__name__} is not a sequence of ids"
    items = list(values)
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            return None, f"an element is {type(item).__name__}"
        if int(item) < 0:
            return None, "an id is negative"
        if int(item) > _INT64_MAX:
            return None, "an id does not fit in int64"
    return np.array(items, dtype=np.int64), None


def _mapping_keys_match(name: str, value: Any, points: Sequence[str]) -> str | None:
    if not isinstance(value, Mapping):
        return f"{name} is not a mapping"
    if set(value) != set(points):
        return f"{name} keys differ from points"
    return None


def check_capture(capture: Any) -> CheckResult:
    """V-0 .. V-7 on ``capture``. A bad capture does not raise; the problems say what is wrong."""
    result = CheckResult()
    problems = result.problems

    def bad(code: str, detail: str) -> None:
        problems.append(Problem(code, detail))

    # V-0 -- the ids, and the attributes the later checks read.
    for name in CAPTURE_ATTRIBUTES:
        if getattr(capture, name, _MISSING) is _MISSING:
            bad(CAPTURE_INVALID, f"V-0 attribute {name} is missing")
    provenance = getattr(capture, "provenance", None)
    if provenance is None and getattr(capture, "provenance", _MISSING) is not _MISSING:
        bad(CAPTURE_INVALID, "V-0 provenance is None")
    if provenance is not None:
        for name in PROVENANCE_ATTRIBUTES:
            if getattr(provenance, name, _MISSING) is _MISSING:
                bad(CAPTURE_INVALID, f"V-0 provenance.{name} is missing")
    if problems:
        return result
    prompt_ids, why = _as_ids(capture.prompt_ids)
    if why is not None:
        bad(CAPTURE_INVALID, f"V-0 prompt_ids: {why}")
    token_ids, why = _as_ids(capture.token_ids)
    if why is not None:
        bad(CAPTURE_INVALID, f"V-0 token_ids: {why}")
    if prompt_ids is not None and len(prompt_ids) < 1:
        bad(CAPTURE_INVALID, "V-0 n_in is 0 (need 1 or more)")
    if problems:
        return result
    result.prompt_ids, result.token_ids = prompt_ids, token_ids
    n_in, n_gen = len(prompt_ids), len(token_ids)
    declared = getattr(capture, "n_in", None)
    if declared is not None and declared != n_in:
        bad(CAPTURE_INVALID, f"V-0 n_in attribute {declared!r} differs from len(prompt_ids)")
    declared = getattr(capture, "n_gen", None)
    if declared is not None and declared != n_gen:
        bad(CAPTURE_INVALID, f"V-0 n_gen attribute {declared!r} differs from len(token_ids)")
    if problems:
        return result
    if n_gen == 0:
        result.empty = True
        return result

    # V-7 -- ranges (the sizes of the RAW header).
    dimension = capture.dimension
    dimension_ok = type(dimension) is int and 1 <= dimension <= MAX_DIMENSION
    if n_gen > MAX_STEPS:
        bad(STEPS_OUT_OF_RANGE, f"V-7 n_gen {n_gen} is over {MAX_STEPS}")
    if not dimension_ok:
        bad(DIMENSION_OUT_OF_RANGE, f"V-7 dimension {dimension!r} is not an int in 1..65535")

    # V-1 -- the points and the mappings keyed by them.
    points = capture.points
    keys_ok = False
    if not isinstance(points, (tuple, list)) or not points:
        bad(CAPTURE_INVALID, "V-1 points is empty or not a sequence")
    elif not all(isinstance(p, str) for p in points):
        bad(CAPTURE_INVALID, "V-1 a point name is not a str")
    elif len(set(points)) != len(points):
        bad(CAPTURE_INVALID, "V-1 points has a duplicate")
    else:
        keys_ok = True
        for name in ("states", "row_rule", "row_pos", "row_decides", "final_norm_applied"):
            why = _mapping_keys_match(name, getattr(capture, name), points)
            if why is not None:
                keys_ok = False
                bad(CAPTURE_INVALID, f"V-1 {why}")
    if not keys_ok:
        return result

    expected_pos = np.arange(n_in - 1, n_in - 1 + n_gen, dtype=np.int64)
    expected_decides = np.arange(n_gen, dtype=np.int64)
    for point in points:
        # V-2 -- the states.
        states = capture.states[point]
        if not isinstance(states, np.ndarray) or states.dtype != np.float32:
            bad(CAPTURE_INVALID, f"V-2 states[{point}] is not a float32 array")
        elif (
            states.ndim != 2
            or states.shape[0] != n_gen
            or (dimension_ok and states.shape[1] != dimension)
        ):
            bad(CAPTURE_INVALID, f"V-2 states[{point}] has shape {states.shape}")
        elif not bool(np.isfinite(states).all()):
            bad(CAPTURE_INVALID, f"V-2 states[{point}] has a value that is not finite")
        # V-3 -- the rows are the ones decides-next/1 says.
        for name, expected in (("row_pos", expected_pos), ("row_decides", expected_decides)):
            found = getattr(capture, name)[point]
            array = found if isinstance(found, np.ndarray) else None
            if (
                array is None
                or array.ndim != 1
                or array.dtype.kind not in "iu"
                or not np.array_equal(array, expected)
            ):
                bad(CAPTURE_INVALID, f"V-3 {name}[{point}] is not {name} of decides-next/1")
        if type(capture.final_norm_applied[point]) is not bool:
            bad(CAPTURE_INVALID, f"V-1 final_norm_applied[{point}] is not a bool")
        if not isinstance(capture.row_rule[point], str):
            bad(CAPTURE_INVALID, f"V-1 row_rule[{point}] is not a str")

    provenance = capture.provenance
    # V-5 -- the hashes of the ids.
    if provenance.prompt_ids_sha256 != ids_sha256(prompt_ids):
        bad(CAPTURE_INVALID, "V-5 prompt_ids_sha256 differs from the hash of prompt_ids")
    if provenance.gen_token_ids_sha256 != ids_sha256(token_ids):
        bad(CAPTURE_INVALID, "V-5 gen_token_ids_sha256 differs from the hash of token_ids")
    # V-6 -- no stop id among the generated ids.
    eos_ids = provenance.eos_ids
    if eos_ids is not None:
        try:
            stops = {int(i) for i in eos_ids}
        except (TypeError, ValueError):
            bad(CAPTURE_INVALID, "V-6 eos_ids is not a sequence of ints")
        else:
            if stops & set(token_ids.tolist()):
                bad(CAPTURE_INVALID, "V-6 an EOS id is among the generated token ids")
    return result

"""The capture vocabulary shared by the observation side and the capture record.

Standard library only. ``record.py`` re-exports these names. The K statuses and reasons
stay in ``record.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

REASON_EMPTY_GENERATION: Final = "empty_generation"

#: The reasons a capture is refused before any contract is made (design U6-5 v0.5, 6.1).
REASON_STEPS_OUT_OF_RANGE: Final = "steps_out_of_range"
REASON_DIMENSION_OUT_OF_RANGE: Final = "dimension_out_of_range"
REASON_CAPTURE_INVALID: Final = "capture_invalid"
REASON_SPACE_COLLISION: Final = "space_collision"
REASON_ORIGIN_MISMATCH: Final = "origin_mismatch"
REASON_ORIGIN_FACTS_INVALID: Final = "origin_facts_invalid"
REASON_CONTRACT_NOT_BUILDABLE: Final = "contract_not_buildable"
REASON_EDGE_UNAVAILABLE: Final = "edge_unavailable"
REASON_EDGE_INCOMPATIBLE: Final = "edge_incompatible"
REASON_CAPTURE_EXTRA_OVERFLOW: Final = "capture_extra_overflow"

CAPTURE_KEYS: Final = (
    "n_in",
    "n_gen",
    "finish_reason",
    "points",
    "model_build_id",
    "compute_environment",
    "adapter",
    "prompt_ids_sha256",
    "gen_token_ids_sha256",
)


def capture_block(
    *,
    n_in: Any,
    n_gen: Any,
    finish_reason: Any,
    points: Sequence[str],
    model_build_id: Any,
    compute_environment: Any,
    adapter: Any,
    prompt_ids_sha256: Any,
    gen_token_ids_sha256: Any,
) -> dict[str, Any]:
    """The ``capture`` part of a record: counts, names and hashes, as they were reported."""
    return {
        "n_in": n_in,
        "n_gen": n_gen,
        "finish_reason": finish_reason,
        "points": list(points),
        "model_build_id": model_build_id,
        "compute_environment": compute_environment,
        "adapter": adapter,
        "prompt_ids_sha256": prompt_ids_sha256,
        "gen_token_ids_sha256": gen_token_ids_sha256,
    }

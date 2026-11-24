"""RawObservationContract v2: verification, the hash of a document and ``contract_facts``.

The contract document stays where it was captured. This module has it verified by the contract
checker of the ``nvs_edge`` package and copies the verified fields into ``contract_facts``
(version ``nvs.contract-facts/1``), the only thing about the contract that a Gate is given. The Gate
never reads the document and never computes its hash.

Nothing is reimplemented here: the structure check, the canonical bytes and ``contract_sha256``
are ``nvs_edge``'s. ``nvs_edge`` is an optional dependency, and an installed ``nvs_edge`` may
predate contract v2. In both cases no ``contract_facts`` is made and the contract binding is
reported as unverifiable -- it is not a failure of the artifact.

The names read from ``nvs_edge`` are listed in ``kernel-api-registry.json``.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

FACTS_VERSION: Final = "nvs.contract-facts/1"

VERIFIED: Final = "VERIFIED"
NOT_VERIFIED: Final = "NOT_VERIFIED"
REJECTED: Final = "REJECTED"

#: Where ``ContractCheck.contract_sha256`` comes from.
HASH_RECOMPUTED_MATCHES_RECORDED: Final = "recomputed_matches_recorded"
HASH_RECOMPUTED_NO_RECORDED: Final = "recomputed_no_recorded"
HASH_RECORDED_ONLY: Final = "recorded_only"
HASH_NONE: Final = "none"

#: Why contract v2 of ``nvs_edge`` cannot be used.
EDGE_NOT_INSTALLED: Final = "nvs_edge_not_installed"
CONTRACT_V2_NOT_AVAILABLE: Final = "contract_v2_not_available"

REASON_STRUCTURE: Final = "contract_structure"
REASON_HASH_MISMATCH: Final = "contract_sha256_mismatch"
REASON_RECORDED_MALFORMED: Final = "recorded_sha256_malformed"

_FAIL: Final = "FAIL"
_HEX64: Final = re.compile(r"[0-9a-f]{64}")
#: What reading a document that is not shaped like a contract can raise.
_READ_ERRORS: Final = (KeyError, TypeError, AttributeError, ValueError, IndexError)

_BINDING_KEYS: Final = (
    "alignment",
    "n_in",
    "n_gen",
    "finish_reason",
    "tokenizer_sha256",
    "chat_template_sha256",
    "prompt_ids_sha256",
    "gen_token_ids_sha256",
)
_ORIGIN_KEYS: Final = ("model_build_id", "compute_environment", "weights_sha256")
_POSITION_KEYS: Final = ("trajectory_id", "row_rule", "dimension", "final_norm_applied")
_STORED_RANGE_KEYS: Final = ("steps", "first_row_pos")
_SPACE_KEYS: Final = (
    "model_build_id",
    "layer",
    "row_rule",
    "compute_environment",
    "dimension",
    "space_rule",
)


class ContractRejected(Exception):
    """The document does not pass the structure check: no ``contract_facts`` for it."""

    code = "CONTRACT_REJECTED"


class ContractUnavailable(Exception):
    """Contract v2 of ``nvs_edge`` cannot be used here (``reason`` says why)."""

    code = "CONTRACT_UNAVAILABLE"

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class EdgeContractApi:
    """The entries of ``nvs_edge`` used here, and the error its structure check raises."""

    validate: Callable[[Mapping[str, Any]], None]
    contract_sha256: Callable[[Mapping[str, Any]], str]
    verify: Callable[..., dict[str, Any]]
    contract_error: type[Exception]
    decode_raw: Callable[..., Any] | None = None


def _load_edge(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> tuple[EdgeContractApi | None, str | None]:
    """``(api, None)``, or ``(None, reason)`` when ``nvs_edge`` or its contract v2 is missing."""
    try:
        import_module("nvs_edge")
    except ImportError:
        return None, EDGE_NOT_INSTALLED
    try:
        modules: dict[str, Any] = {}
        for name in ("contract", "contract_v2", "verify_contract_v2"):
            modules[name] = import_module(f"nvs_edge.{name}")
        raw_store = import_module("nvs_edge.raw_store")
        wanted = (
            ("contract", "ContractError"),
            ("contract", "contract_sha256"),
            ("contract_v2", "validate_contract_v2"),
            ("verify_contract_v2", "verify_contract_v2"),
        )
        for module, attribute in wanted:
            if not hasattr(modules[module], attribute):
                return None, CONTRACT_V2_NOT_AVAILABLE

        def read(module: str, attribute: str) -> Any:
            return getattr(modules[module], attribute)

        api = EdgeContractApi(
            validate=read("contract_v2", "validate_contract_v2"),
            contract_sha256=read("contract", "contract_sha256"),
            verify=read("verify_contract_v2", "verify_contract_v2"),
            contract_error=read("contract", "ContractError"),
            decode_raw=getattr(raw_store, "decode_raw", None),
        )
    except ImportError:
        return None, CONTRACT_V2_NOT_AVAILABLE
    return api, None


_EDGE, _EDGE_UNAVAILABLE = _load_edge()


def edge_contract_available() -> bool:
    """Whether contract v2 of ``nvs_edge`` can be used in this process."""
    return _EDGE is not None


def recompute_contract_sha256(document: Mapping[str, Any]) -> str:
    """The ``contract_sha256`` of ``document``, as ``nvs_edge`` computes it.

    Raises ``ContractUnavailable`` when contract v2 of ``nvs_edge`` cannot be used here.
    """
    edge = _EDGE
    if edge is None:
        raise ContractUnavailable(_EDGE_UNAVAILABLE or EDGE_NOT_INSTALLED)
    return edge.contract_sha256(document)


@dataclass(frozen=True, slots=True)
class ContractCheck:
    """What is known about one contract document before an envelope is built from it.

    ``status``: ``VERIFIED`` (structure passed; the hash is ``nvs_edge``'s), ``NOT_VERIFIED``
    (contract v2 of ``nvs_edge`` is not available here) or ``REJECTED`` (structure failed, or the
    recorded hash differs). ``facts`` is set only for ``VERIFIED``. ``hash_source`` says whether
    ``contract_sha256`` was compared with a recorded value. ``synthetic`` is ``origin.synthetic``
    when it is a bool, else ``None``. ``report`` is the report of ``verify_contract_v2`` as it is.
    """

    status: str
    contract_sha256: str | None
    hash_source: str
    facts: dict[str, Any] | None
    synthetic: bool | None
    reasons: tuple[str, ...]
    report: dict[str, Any] | None


def _synthetic_of(document: Any) -> bool | None:
    origin = document.get("origin") if isinstance(document, Mapping) else None
    value = origin.get("synthetic") if isinstance(origin, Mapping) else None
    return value if isinstance(value, bool) else None


def _facts(document: Mapping[str, Any], contract_sha256: str) -> dict[str, Any]:
    """The fields of a structurally valid document, copied as they are (no value is changed)."""
    header, binding, origin = document["contract"], document["binding"], document["origin"]
    positions: dict[str, Any] = {}
    for position in document["positions"]:
        entry: dict[str, Any] = {key: position[key] for key in _POSITION_KEYS}
        entry["stored_range"] = {key: position["stored_range"][key] for key in _STORED_RANGE_KEYS}
        entry["space"] = {key: position["space"][key] for key in _SPACE_KEYS}
        positions[position["point"]] = entry
    return {
        "facts_version": FACTS_VERSION,
        "contract": {
            "name": header["name"],
            "version": header["version"],
            "contract_sha256": contract_sha256,
        },
        "synthetic": origin["synthetic"],
        "binding": {key: binding[key] for key in _BINDING_KEYS},
        "origin": {key: origin[key] for key in _ORIGIN_KEYS},
        "positions": positions,
    }


def contract_facts_from_v2(document: Mapping[str, Any]) -> dict[str, Any] | None:
    """``contract_facts`` of a RawObservationContract v2 document, or ``None``.

    ``None`` when contract v2 of ``nvs_edge`` is not available: the Gate is then given no facts and
    reports ``contract_not_supplied``. ``ContractRejected`` when the document does not pass
    ``validate_contract_v2``: facts are never made from a document that was not checked.
    The result shares no mapping with ``document``.
    """
    edge = _EDGE
    if edge is None:
        return None
    try:
        edge.validate(document)
        digest = edge.contract_sha256(document)
    except edge.contract_error as exc:
        raise ContractRejected(f"contract structure: {exc}") from exc
    except _READ_ERRORS as exc:
        raise ContractRejected(f"contract unreadable: {exc!r}") from exc
    return _facts(document, digest)


def rows_from_raw(raw_bytes: Mapping[str, bytes] | None) -> dict[str, bytes]:
    """point -> the rows of its RAW file as bytes (float32, little-endian, C order).

    Decoding is ``nvs_edge``'s (``raw_store.decode_raw``, which checks the file against its own
    ``trajectory_id`` and tail hash). A file that cannot be decoded is left out -- the
    verification of the same file reports it -- and without ``nvs_edge`` the result is empty.
    """
    edge = _EDGE
    if edge is None or edge.decode_raw is None or raw_bytes is None:
        return {}
    out: dict[str, bytes] = {}
    for point, data in raw_bytes.items():
        try:
            states, _header = edge.decode_raw(data, dimension=None)
            out[point] = states.astype("<f4").tobytes()
        except Exception:  # noqa: BLE001 -- RawStoreError, and the zstd error of a damaged body
            continue
    return out


def check_contract_v2(
    document: Mapping[str, Any],
    *,
    recorded_sha256: str | None = None,
    raw_bytes: Mapping[str, bytes] | None = None,
    instrument_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> ContractCheck:
    """Have ``document`` verified by ``nvs_edge`` and say what may be built from it. Does not raise.

    ``recorded_sha256`` is the hash written when the contract was made; without it the contract
    can still be ``VERIFIED`` and ``hash_source`` records that nothing was compared. ``raw_bytes``
    and ``instrument_records`` (keyed by point name) are passed on; the per-position results are
    in ``report`` and do not decide ``status``.
    """
    synthetic = _synthetic_of(document)
    edge = _EDGE
    if edge is None:
        reasons = [_EDGE_UNAVAILABLE or EDGE_NOT_INSTALLED]
        recorded = recorded_sha256
        if recorded is not None and not (
            isinstance(recorded, str) and _HEX64.fullmatch(recorded)
        ):
            reasons.append(REASON_RECORDED_MALFORMED)
            recorded = None
        return ContractCheck(
            status=NOT_VERIFIED,
            contract_sha256=recorded,
            hash_source=HASH_NONE if recorded is None else HASH_RECORDED_ONLY,
            facts=None,
            synthetic=synthetic,
            reasons=tuple(reasons),
            report=None,
        )
    report = edge.verify(
        document,
        expected_contract_sha256=recorded_sha256,
        raw_bytes=raw_bytes,
        instrument_records=instrument_records,
    )
    checks = report["G1"]["checks"]
    rejected = []
    if checks.get("contract_structure") == _FAIL:
        rejected.append(REASON_STRUCTURE)
    if checks.get("contract_sha256") == _FAIL:
        rejected.append(REASON_HASH_MISMATCH)
    facts = None
    if not rejected:
        try:
            facts = contract_facts_from_v2(document)
        except ContractRejected:  # the report passed a structure this process then refused
            rejected.append(REASON_STRUCTURE)
    if facts is None:
        return ContractCheck(
            status=REJECTED,
            contract_sha256=None,
            hash_source=HASH_NONE,
            facts=None,
            synthetic=synthetic,
            reasons=tuple(rejected),
            report=report,
        )
    return ContractCheck(
        status=VERIFIED,
        contract_sha256=facts["contract"]["contract_sha256"],
        hash_source=(
            HASH_RECOMPUTED_NO_RECORDED
            if recorded_sha256 is None
            else HASH_RECOMPUTED_MATCHES_RECORDED
        ),
        facts=facts,
        synthetic=synthetic,
        reasons=(),
        report=report,
    )

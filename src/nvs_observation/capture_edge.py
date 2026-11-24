"""The ``nvs_edge`` entries that U6-5 uses, loaded the way ``contract._load_edge`` loads its own
(design U6-5 v0.4, 8).

``nvs_edge`` is an optional dependency. Any name that cannot be read is ``edge_unavailable``
(the name is given); a version that does not fit is ``edge_incompatible``. The module is loaded
once, when this module is imported. ``contract.py`` is not changed and not shared.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from nvs_observation.contracts.capture_record import (
    REASON_EDGE_INCOMPATIBLE,
    REASON_EDGE_UNAVAILABLE,
)
from nvs_observation.contracts.constants import MAX_DIMENSION, POINT_PATTERN

EDGE_UNAVAILABLE: Final = REASON_EDGE_UNAVAILABLE
EDGE_INCOMPATIBLE: Final = REASON_EDGE_INCOMPATIBLE

_EXPECTED_CONTRACT_VERSION: Final = 2
_EXPECTED_INSTRUMENT_NAME: Final = "raw-nd-v1"
_EXPECTED_INSTRUMENT_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class CaptureEdge:
    """The entries of ``nvs_edge`` that U6-5 calls."""

    build_contract_v2: Callable[..., dict[str, Any]]
    unknown2: Callable[..., Any]
    contract_error: type[Exception]
    layer_label: Callable[[str], int | str]
    point_pattern: Any
    space_rule: str
    instrument_record: Callable[..., dict[str, Any]]
    encode_raw: Callable[..., tuple[bytes, Any]]
    decode_raw: Callable[..., tuple[Any, Any]]
    instrument_name: str
    instrument_version: int
    instrument_settings_sha256: Callable[[], str]
    source_build_id: Callable[[], str]


@dataclass(frozen=True, slots=True)
class EdgeFailure:
    """``code`` is ``edge_unavailable`` or ``edge_incompatible``; ``detail`` names what is wrong."""

    code: str
    detail: str


def load_capture_edge(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> tuple[CaptureEdge | None, EdgeFailure | None]:
    """``(edge, None)``, or ``(None, failure)``. Does not raise."""
    try:
        import_module("nvs_edge")
    except ImportError:
        return None, EdgeFailure(EDGE_UNAVAILABLE, "nvs_edge")
    modules: dict[str, Any] = {}
    for name in ("contract", "contract_v2", "raw_store", "raw_nd", "client"):
        try:
            modules[name] = import_module(f"nvs_edge.{name}")
        except ImportError:
            return None, EdgeFailure(EDGE_UNAVAILABLE, f"nvs_edge.{name}")

    def read(module: str, attribute: str) -> Any:
        return getattr(modules[module], attribute)

    wanted = (
        ("contract", "ContractError"),
        ("contract_v2", "build_contract_v2"),
        ("contract_v2", "Unknown2"),
        ("contract_v2", "layer_label"),
        ("contract_v2", "POINT_PATTERN"),
        ("contract_v2", "SPACE_RULE"),
        ("contract_v2", "CONTRACT_VERSION_V2"),
        ("contract_v2", "MAX_DIMENSION"),
        ("contract_v2", "instrument_record"),
        ("raw_store", "encode_raw"),
        ("raw_store", "decode_raw"),
        ("raw_nd", "NAME"),
        ("raw_nd", "VERSION"),
        ("raw_nd", "settings_sha256"),
        ("client", "source_build_id"),
    )
    for module, attribute in wanted:
        if not hasattr(modules[module], attribute):
            return None, EdgeFailure(EDGE_UNAVAILABLE, f"nvs_edge.{module}.{attribute}")
    # The version, dimension and instrument values of nvs_edge are compared with the ones
    # this package expects. The names read from nvs_edge are listed in
    # kernel-api-registry.json.
    for module, attribute, expected in (
        ("contract_v2", "CONTRACT_VERSION_V2", _EXPECTED_CONTRACT_VERSION),
        ("contract_v2", "MAX_DIMENSION", MAX_DIMENSION),
        ("raw_nd", "NAME", _EXPECTED_INSTRUMENT_NAME),
        ("raw_nd", "VERSION", _EXPECTED_INSTRUMENT_VERSION),
    ):
        found = read(module, attribute)
        if found != expected or isinstance(found, bool):
            return None, EdgeFailure(
                EDGE_INCOMPATIBLE, f"nvs_edge.{module}.{attribute} is {found!r}, need {expected!r}"
            )
    return (
        CaptureEdge(
            build_contract_v2=read("contract_v2", "build_contract_v2"),
            unknown2=read("contract_v2", "Unknown2"),
            contract_error=read("contract", "ContractError"),
            layer_label=read("contract_v2", "layer_label"),
            point_pattern=POINT_PATTERN,
            space_rule=read("contract_v2", "SPACE_RULE"),
            instrument_record=read("contract_v2", "instrument_record"),
            encode_raw=read("raw_store", "encode_raw"),
            decode_raw=read("raw_store", "decode_raw"),
            instrument_name=read("raw_nd", "NAME"),
            instrument_version=read("raw_nd", "VERSION"),
            instrument_settings_sha256=read("raw_nd", "settings_sha256"),
            source_build_id=read("client", "source_build_id"),
        ),
        None,
    )


_EDGE, _FAILURE = load_capture_edge()


def current() -> tuple[CaptureEdge | None, EdgeFailure | None]:
    """What this process loaded (read again each call, so a test can replace it)."""
    return _EDGE, _FAILURE

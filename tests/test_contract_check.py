"""contract_check: ``check_contract_v2`` and ``contract_facts_from_v2``, with a stand-in for the
entries of ``nvs_edge`` (tests T-U6-C03..C09 of the contract design). They need neither the
kernel nor any other package: this module's own decisions are exercised everywhere. The same
checks against the real ``nvs_edge`` are in ``tests_private/test_contract_check_real.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nvs_observation import contract_check as contract_module
from nvs_observation.contract_check import (
    CONTRACT_V2_NOT_AVAILABLE,
    EDGE_NOT_INSTALLED,
    HASH_NONE,
    HASH_RECOMPUTED_MATCHES_RECORDED,
    HASH_RECOMPUTED_NO_RECORDED,
    HASH_RECORDED_ONLY,
    NOT_VERIFIED,
    REJECTED,
    VERIFIED,
    ContractRejected,
    EdgeContractApi,
    _load_edge,
    check_contract_v2,
    contract_facts_from_v2,
    edge_contract_available,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = json.loads((FIXTURES / "contract-v2-sample.json").read_text(encoding="utf-8"))
EXPECTED_FACTS = json.loads((FIXTURES / "contract-facts-sample.json").read_text(encoding="utf-8"))[
    "facts"
]
RECORDED = SAMPLE["contract_sha256"]


def document() -> dict[str, Any]:
    return copy.deepcopy(SAMPLE["document"])


class FakeContractError(Exception):
    pass


def _fake_sha(doc: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _fake_validate(doc: Mapping[str, Any]) -> None:
    if doc.get("session_id") == "broken":
        raise FakeContractError("session broken")
    if not isinstance(doc.get("positions"), list):
        raise TypeError("positions")


def _fake_verify(doc: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    try:
        _fake_validate(doc)
        structure = "PASS"
    except (FakeContractError, TypeError):
        structure = "FAIL"
    expected = kwargs["expected_contract_sha256"]
    if expected is None:
        digest = "UNVERIFIABLE"
    else:
        digest = "PASS" if expected == _fake_sha(doc) else "FAIL"
    result = "FAIL" if "FAIL" in (structure, digest) else "UNVERIFIABLE"
    return {
        "G1": {
            "result": result,
            "checks": {"contract_structure": structure, "contract_sha256": digest},
        },
        "passed_on": {key: kwargs[key] for key in ("raw_bytes", "instrument_records")},
    }


FAKE = EdgeContractApi(
    validate=_fake_validate,
    contract_sha256=_fake_sha,
    verify=_fake_verify,
    contract_error=FakeContractError,
)


@pytest.fixture
def fake_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract_module, "_EDGE", FAKE)
    monkeypatch.setattr(contract_module, "_EDGE_UNAVAILABLE", None)


@pytest.fixture
def no_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract_module, "_EDGE", None)
    monkeypatch.setattr(contract_module, "_EDGE_UNAVAILABLE", EDGE_NOT_INSTALLED)


def test_stand_in_structure_failure(fake_edge: None) -> None:
    doc = document()
    doc["session_id"] = "broken"
    with pytest.raises(ContractRejected, match="contract structure: session broken"):
        contract_facts_from_v2(doc)
    check = check_contract_v2(doc)
    assert (check.status, check.facts, check.reasons) == (REJECTED, None, ("contract_structure",))


def test_stand_in_unreadable_document(fake_edge: None) -> None:
    doc = document()
    doc["positions"] = "not a list"
    with pytest.raises(ContractRejected, match="contract unreadable"):
        contract_facts_from_v2(doc)
    assert check_contract_v2(doc).status == REJECTED


def test_report_that_passed_what_validation_refuses(
    fake_edge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Facts come from this process's own validation, never from a report alone."""
    lenient = EdgeContractApi(
        validate=_fake_validate,
        contract_sha256=_fake_sha,
        verify=lambda doc, **kwargs: {"G1": {"checks": {"contract_structure": "PASS"}}},
        contract_error=FakeContractError,
    )
    monkeypatch.setattr(contract_module, "_EDGE", lenient)
    doc = document()
    doc["session_id"] = "broken"
    check = check_contract_v2(doc)
    assert (check.status, check.facts, check.reasons) == (REJECTED, None, ("contract_structure",))


def test_stand_in_recorded_hash_mismatch(fake_edge: None) -> None:
    check = check_contract_v2(document(), recorded_sha256="0" * 64)
    assert (check.status, check.reasons) == (REJECTED, ("contract_sha256_mismatch",))
    assert check.facts is None


def test_stand_in_hash_sources_and_arguments(fake_edge: None) -> None:
    doc = document()
    raw, records = {"final_norm": b"x"}, {"final_norm": {"output": {}}}
    check = check_contract_v2(doc, raw_bytes=raw, instrument_records=records)
    assert (check.status, check.hash_source) == (VERIFIED, HASH_RECOMPUTED_NO_RECORDED)
    assert check.contract_sha256 == _fake_sha(doc)
    assert check.report is not None
    assert check.report["passed_on"] == {"raw_bytes": raw, "instrument_records": records}
    matched = check_contract_v2(doc, recorded_sha256=_fake_sha(doc))
    assert (matched.status, matched.hash_source) == (VERIFIED, HASH_RECOMPUTED_MATCHES_RECORDED)
    assert matched.facts is not None
    assert matched.facts["contract"]["contract_sha256"] == _fake_sha(doc)


def test_stand_in_copies_only_the_listed_fields(fake_edge: None) -> None:
    doc = document()
    doc["origin"]["weights_sha256"] = "unknown"
    facts = contract_facts_from_v2(doc)
    assert facts is not None
    assert set(facts) == {
        "facts_version",
        "contract",
        "synthetic",
        "binding",
        "origin",
        "positions",
    }
    assert set(facts["binding"]) == set(EXPECTED_FACTS["binding"])
    assert facts["origin"] == {
        "model_build_id": doc["origin"]["model_build_id"],
        "compute_environment": doc["origin"]["compute_environment"],
        "weights_sha256": "unknown",
    }
    assert list(facts["positions"]) == [p["point"] for p in doc["positions"]]
    for position in doc["positions"]:
        entry = facts["positions"][position["point"]]
        assert set(entry) == {
            "trajectory_id",
            "row_rule",
            "dimension",
            "final_norm_applied",
            "stored_range",
            "space",
        }
        assert set(entry["stored_range"]) == {"steps", "first_row_pos"}
        assert entry["space"] == position["space"]
        assert type(entry["space"]["layer"]) is type(position["space"]["layer"])


def _importer(missing: set[str]) -> Any:
    def import_module(name: str) -> Any:
        if name in missing:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return type(sys)(name)  # an empty module: no attribute of the real one

    return import_module


def test_load_edge_tells_the_two_reasons_apart() -> None:
    assert _load_edge(_importer({"nvs_edge"})) == (None, EDGE_NOT_INSTALLED)
    assert _load_edge(_importer({"nvs_edge.contract_v2"})) == (None, CONTRACT_V2_NOT_AVAILABLE)
    assert _load_edge(_importer({"nvs_edge.verify_contract_v2"})) == (
        None,
        CONTRACT_V2_NOT_AVAILABLE,
    )
    # every module imports but an entry is missing (an nvs_edge of another shape)
    assert _load_edge(_importer(set())) == (None, CONTRACT_V2_NOT_AVAILABLE)


def test_load_edge_returns_the_entries_when_they_exist() -> None:
    modules = {name: type(sys)(name) for name in ("nvs_edge.contract", "nvs_edge.contract_v2")}
    modules["nvs_edge.verify_contract_v2"] = type(sys)("nvs_edge.verify_contract_v2")
    modules["nvs_edge.contract"].contract_sha256 = _fake_sha  # type: ignore[attr-defined]
    modules["nvs_edge.contract"].ContractError = FakeContractError  # type: ignore[attr-defined]
    modules["nvs_edge.contract_v2"].validate_contract_v2 = _fake_validate  # type: ignore[attr-defined]
    modules["nvs_edge.verify_contract_v2"].verify_contract_v2 = _fake_verify  # type: ignore[attr-defined]
    api, reason = _load_edge(lambda name: modules.get(name, type(sys)(name)))
    assert reason is None and api == FAKE


def test_without_nvs_edge_no_facts_and_no_exception(no_edge: None) -> None:
    assert edge_contract_available() is False
    assert contract_facts_from_v2(document()) is None
    assert contract_facts_from_v2({"not": "a contract"}) is None
    check = check_contract_v2(document(), recorded_sha256=RECORDED)
    assert check.status == NOT_VERIFIED
    assert check.reasons == (EDGE_NOT_INSTALLED,)
    assert check.facts is None and check.report is None
    assert (check.contract_sha256, check.hash_source) == (RECORDED, HASH_RECORDED_ONLY)
    assert check.synthetic is False


def test_without_nvs_edge_and_without_a_recorded_hash(no_edge: None) -> None:
    check = check_contract_v2(document())
    assert check.status == NOT_VERIFIED
    assert (check.contract_sha256, check.hash_source) == (None, HASH_NONE)


@pytest.mark.parametrize("recorded", ["abc", "A" * 64, 5, ""])
def test_without_nvs_edge_a_malformed_recorded_hash_is_not_used(
    no_edge: None, recorded: Any
) -> None:
    check = check_contract_v2(document(), recorded_sha256=recorded)
    assert check.status == NOT_VERIFIED
    assert check.reasons == (EDGE_NOT_INSTALLED, "recorded_sha256_malformed")
    assert (check.contract_sha256, check.hash_source) == (None, HASH_NONE)


def test_contract_missing_is_reported_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract_module, "_EDGE", None)
    monkeypatch.setattr(contract_module, "_EDGE_UNAVAILABLE", CONTRACT_V2_NOT_AVAILABLE)
    assert check_contract_v2(document()).reasons == (CONTRACT_V2_NOT_AVAILABLE,)


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ({"synthetic": True}, True),
        ({"synthetic": False}, False),
        ({"synthetic": "false"}, None),
        ({"synthetic": 0}, None),
        ({}, None),
        ("unknown", None),
    ],
)
def test_synthetic_is_reported_only_when_it_is_a_bool(
    no_edge: None, origin: Any, expected: bool | None
) -> None:
    assert check_contract_v2({"origin": origin}).synthetic is expected


def test_changing_the_facts_does_not_change_the_document(fake_edge: None) -> None:
    doc = document()
    before = copy.deepcopy(doc)
    facts = contract_facts_from_v2(doc)
    assert facts is not None
    point = doc["positions"][0]["point"]
    facts["binding"]["n_in"] = 999
    facts["origin"]["model_build_id"] = "changed"
    facts["contract"]["version"] = 3
    facts["positions"][point]["space"]["layer"] = "changed"
    facts["positions"][point]["stored_range"]["steps"] = 999
    facts["positions"][point]["row_rule"] = "changed"
    del facts["positions"][doc["positions"][1]["point"]]
    assert doc == before


# ---- rows_from_raw --------------------------------------------------------------------------


def _rows() -> bytes:
    return struct.pack("<8f", *[float(i) for i in range(8)])


class _States:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def astype(self, dtype: str) -> _States:
        assert dtype == "<f4"
        return self

    def tobytes(self) -> bytes:
        return self.data


def _fake_decode(data: bytes, *, dimension: int | None) -> tuple[_States, None]:
    assert dimension is None
    if not data.startswith(b"RAW:"):
        raise RuntimeError("not a RAW file")
    return _States(data[4:]), None


def _api(decode: Any) -> EdgeContractApi:
    def unused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("not used by rows_from_raw")

    return EdgeContractApi(
        validate=unused,
        contract_sha256=unused,
        verify=unused,
        contract_error=ValueError,
        decode_raw=decode,
    )


def test_rows_from_raw_leaves_out_what_cannot_be_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract_module, "_EDGE", _api(_fake_decode))
    raw = {"final_norm": b"RAW:" + _rows(), "block_out:3": b"damaged"}
    assert contract_module.rows_from_raw(raw) == {"final_norm": _rows()}
    assert contract_module.rows_from_raw(None) == {}
    assert contract_module.rows_from_raw({}) == {}


def test_rows_from_raw_without_nvs_edge_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {"final_norm": b"RAW:" + _rows()}
    monkeypatch.setattr(contract_module, "_EDGE", None)
    assert contract_module.rows_from_raw(raw) == {}
    monkeypatch.setattr(contract_module, "_EDGE", _api(None))
    assert contract_module.rows_from_raw(raw) == {}


def test_the_bare_hash_needs_the_kernel_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract_module, "_EDGE", None)
    monkeypatch.setattr(contract_module, "_EDGE_UNAVAILABLE", EDGE_NOT_INSTALLED)
    with pytest.raises(contract_module.ContractUnavailable) as caught:
        contract_module.recompute_contract_sha256(document())
    assert caught.value.reason == EDGE_NOT_INSTALLED
    monkeypatch.setattr(contract_module, "_EDGE", FAKE)
    assert contract_module.recompute_contract_sha256(document()) == _fake_sha(document())


def test_a_rejected_contract_is_not_an_error_of_the_runtime() -> None:
    assert not any(base.__module__.startswith("nvs_runtime") for base in ContractRejected.__mro__)
    assert ContractRejected.code == "CONTRACT_REJECTED"

"""(b) Kernel Adapter API Registry matches the nvs_edge references in the sources."""

from __future__ import annotations

from _edge_refs import extract_all_nvs_edge_refs, load_registry, registry_ref_set


def test_b_registry_matches_static_nvs_edge_refs() -> None:
    entries = load_registry()
    from_registry = registry_ref_set(entries)
    from_source = extract_all_nvs_edge_refs()
    assert from_source == from_registry, (
        f"only in source: {sorted(from_source - from_registry)!r}; "
        f"only in registry: {sorted(from_registry - from_source)!r}"
    )

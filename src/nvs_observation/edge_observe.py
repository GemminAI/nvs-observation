"""One Edge session's measurement, without the provider registry.

Standard library, plus ``nvs_edge`` when it is installed. Nothing else of ``nvs_runtime``.
``nvs_edge`` is read once, when this module is imported, through ``importlib``.
"""

from __future__ import annotations

import importlib
import secrets
from collections.abc import Callable
from typing import Any, cast


def _load_nvs_edge(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> tuple[Any, type[BaseException], Any]:
    """``Edge``, ``EdgeTransportError``, ``RawCapture``. Missing means nvs_edge is absent."""
    edge_name = "Edge"
    error_name = "EdgeTransportError"
    raw_name = "RawCapture"
    try:
        edge = import_module("nvs_edge.edge")
        raw_store = import_module("nvs_edge.raw_store")
        return (
            getattr(edge, edge_name),
            cast(type[BaseException], getattr(edge, error_name)),
            getattr(raw_store, raw_name),
        )
    except (ImportError, AttributeError):
        return None, Exception, None


_Edge, _EdgeTransportError, _RawCapture = _load_nvs_edge()


def _deferred(reason: str, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "board_record",
        "status": "DEFERRED",
        "reason": reason,
        "request": request,
    }


class EdgeObserver:
    """The sessions an ``EdgeProvider`` measures. The provider's ``_edges`` is ``edges``."""

    def __init__(self) -> None:
        self.raw_captures: dict[str, Any] = {}
        self.session_keys: dict[str, bytes] = {}
        self.session_tiers: dict[str, str] = {}
        self.session_tier_tokens: dict[str, str] = {}
        self.edges: dict[str, Any] = {}
        self.unreachable: set[str] = set()

    def set_session_key(self, session_id: str, key: bytes) -> None:
        if not isinstance(key, bytes) or not key:
            raise ValueError("the session key must be non-empty bytes")
        if session_id in self.edges:
            raise ValueError(
                f"session {session_id!r} has already observed a token; its key is fixed"
            )
        self.session_keys[session_id] = key

    def set_session_tier(self, session_id: str, tier: str) -> None:
        if tier not in ("basic", "pro"):
            raise ValueError(f"tier must be 'basic' or 'pro', got {tier!r}")
        if session_id in self.edges:
            raise ValueError(
                f"session {session_id!r} has already observed a token; its tier is fixed"
            )
        self.session_tiers[session_id] = tier

    def set_session_tier_token(self, session_id: str, tier_token: str | None) -> None:
        if session_id in self.edges:
            raise ValueError(
                f"session {session_id!r} has already observed a token; its tier is fixed"
            )
        if tier_token is None:
            self.session_tier_tokens.pop(session_id, None)
        else:
            self.session_tier_tokens[session_id] = tier_token

    def session_tier(self, session_id: str) -> str | None:
        edge = self.edges.get(session_id)
        return None if edge is None else edge.tier

    def edge_for(
        self,
        session_id: str,
        *,
        base_url: str | None,
        anchor_policy_sha256: str | None,
        experiment_id: str,
        transport: Any,
        wiring_policy: str | None,
        capture_raw: bool,
    ) -> Any:
        if session_id not in self.edges:
            extra: dict[str, Any] = {}
            if session_id in self.session_tier_tokens:
                extra["tier_token"] = self.session_tier_tokens.pop(session_id)
            if session_id in self.session_tiers:
                extra["tier"] = self.session_tiers.pop(session_id)
            if wiring_policy is not None:
                extra["wiring_policy"] = wiring_policy
            if capture_raw and _RawCapture is not None:
                self.raw_captures[session_id] = _RawCapture()
                extra["raw_capture"] = self.raw_captures[session_id]
            self.edges[session_id] = _Edge(
                session_id,
                self.session_keys.pop(session_id, None) or secrets.token_bytes(32),
                anchor_policy_sha256,
                base_url=base_url,
                experiment_id=experiment_id,
                run_id=f"{experiment_id}-{session_id}",
                transport=transport,
                **extra,
            )
        return self.edges[session_id]

    def raw_capture_for(self, session_id: str) -> Any:
        return self.raw_captures.get(session_id)

    def observe(
        self,
        *,
        session_id: str,
        token_index: int,
        hidden_state: Any,
        available: bool,
        base_url: str | None,
        in_process: bool,
        edge_for: Callable[[str], Any],
    ) -> dict[str, Any]:
        """One token. ``available`` is the provider module's ``_NVS_EDGE_AVAILABLE``.
        ``edge_for`` is that provider's ``_edge_for``, so a subclass override measures."""
        request = {"session_id": session_id, "token_index": token_index}
        if not available:
            return _deferred("nvs_edge is not installed in this environment", request)
        if not base_url and not in_process:
            return _deferred("no kernel base_url configured", request)
        if session_id in self.unreachable:
            return _deferred("kernel was unreachable earlier in this session", request)
        edge = edge_for(session_id)
        try:
            record, causal_log = edge.observe_token_full(token_index, hidden_state)
        except _EdgeTransportError as exc:
            self.unreachable.add(session_id)
            return _deferred(f"kernel unreachable: {exc}", request)
        return {
            "type": "board_record",
            "status": "OK",
            "session_id": session_id,
            "token_index": token_index,
            "record": record.to_wire(),
            "causal_log": causal_log,
        }

"""``AdapterSession``: a capture and the ``origin_facts()`` of that capture, as one pair
(design U6-5 v0.4, 3.1).

``origin_facts()`` takes no argument: an Adapter makes it from the capture it made last. Two
callers sharing one Adapter could therefore get the facts of each other's capture, and equal values
would not show it. ``AdapterSession`` holds one lock for the instance and calls ``capture`` and
``origin_facts`` one after the other under it.

Rules for the caller (L-d), which this module cannot enforce: use one Adapter instance through one
``AdapterSession`` only, and never call ``adapter.capture()`` yourself.

This module imports the standard library only and does not import ``sensos_inference``.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

#: ``adapter._last_capture is capture`` (L-e).
LAST_CAPTURE_SAME: Final = "same"
#: ``adapter._last_capture`` exists and is another object: the pair is not one.
LAST_CAPTURE_NOT_LAST: Final = "not_last"
#: The Adapter has no ``_last_capture``: not checked (and said so in ``adapter_notes``).
LAST_CAPTURE_NOT_CHECKABLE: Final = "not_checkable"

_MISSING: Final = object()


@dataclass(frozen=True, slots=True)
class CapturedWithOrigin:
    """One capture and the ``origin_facts()`` made right after it, with the lock held."""

    capture: Any
    origin_facts: Mapping[str, Any]
    adapter_class: str
    sequence: int
    last_capture_check: str


class AdapterSession:
    """Wraps one Adapter instance; ``capture_with_origin`` is the only way to use it (L-d)."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._lock = threading.Lock()
        self._sequence = 0

    @property
    def adapter_class(self) -> str:
        return type(self._adapter).__name__

    def capture_with_origin(
        self,
        messages: Any,
        request: Any,
        *,
        model_build_id: str,
        compute_environment: str,
    ) -> CapturedWithOrigin:
        """``adapter.capture(...)`` then ``adapter.origin_facts()``, with nothing in between.

        An exception from either call is raised as it is (the lock is released).
        """
        with self._lock:
            capture = self._adapter.capture(
                messages,
                request,
                model_build_id=model_build_id,
                compute_environment=compute_environment,
            )
            facts = self._adapter.origin_facts()
            self._sequence += 1
            last = getattr(self._adapter, "_last_capture", _MISSING)
            if last is _MISSING:
                check = LAST_CAPTURE_NOT_CHECKABLE
            else:
                check = LAST_CAPTURE_SAME if last is capture else LAST_CAPTURE_NOT_LAST
            return CapturedWithOrigin(
                capture=capture,
                origin_facts=facts,
                adapter_class=self.adapter_class,
                sequence=self._sequence,
                last_capture_check=check,
            )

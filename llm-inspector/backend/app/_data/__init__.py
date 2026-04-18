"""
app/_data — Provenance Registry

Exposes a single lazy-loaded singleton `SRC` for accessing all registered
constants with full data provenance.

Usage:
    from app._data import SRC

    threshold = SRC["verdict.adv_spoof_cap"].value
    print(SRC["verdict.adv_spoof_cap"].source_url)
"""
from __future__ import annotations

from app._data.sources import SourcesRegistry, get_registry

# Lazy singleton — instantiated on first access, not at import time.
# This avoids slowing down unit tests that don't need the full registry.
class _SRCProxy:
    """Lazy proxy for the SourcesRegistry singleton."""

    _registry: SourcesRegistry | None = None

    def _get(self) -> SourcesRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    def __getitem__(self, key: str):
        return self._get()[key]

    def __contains__(self, key: str) -> bool:
        return key in self._get()

    def all_ids(self) -> list[str]:
        return self._get().all_ids()

    def placeholders(self) -> list[str]:
        """Return ids tagged phase2_replace=true — not yet data-fitted."""
        return self._get().placeholders()


SRC = _SRCProxy()

__all__ = ["SRC", "SourcesRegistry", "get_registry"]

"""
app/_data/sources.py — SourcesRegistry

Loads and validates SOURCES.yaml, providing typed access to every registered
constant with its full data-provenance record.

Design goals:
- Zero hard-coded constants in this module; all values live in SOURCES.yaml.
- SHA-256 content hash cached alongside the registry so mutations are detected.
- Thread-safe singleton via module-level lock.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Any

_SOURCES_PATH = pathlib.Path(__file__).parent / "SOURCES.yaml"
_LOCK = threading.Lock()
_INSTANCE: "SourcesRegistry | None" = None


@dataclass
class SourceEntry:
    """A single provenance record from SOURCES.yaml."""
    id: str
    value: Any
    unit: str
    source_url: str
    source_type: str          # paper | official_doc | dataset | empirical | derived
    retrieved_at: str         # ISO-8601 date string
    license: str
    note: str = ""
    phase2_replace: bool = False  # True → value is a placeholder; Phase 2 will refit

    # Required fields (validated at load time)
    REQUIRED_FIELDS: frozenset[str] = field(default_factory=lambda: frozenset(
        {"id", "value", "unit", "source_url", "source_type", "retrieved_at", "license"}
    ), init=False, repr=False, compare=False)

    def __post_init__(self):
        for f in ("id", "source_url", "source_type", "retrieved_at", "license"):
            if not getattr(self, f):
                raise ValueError(f"SourceEntry '{self.id}': field '{f}' must not be empty.")
        valid_types = {"paper", "official_doc", "dataset", "empirical", "derived"}
        if self.source_type not in valid_types:
            raise ValueError(
                f"SourceEntry '{self.id}': source_type='{self.source_type}' "
                f"not in {valid_types}"
            )


class SourcesRegistry:
    """
    Immutable registry of all SOURCES.yaml entries, keyed by `id`.

    Thread-safe after construction.
    """

    def __init__(self, entries: list[SourceEntry], file_sha256: str) -> None:
        self._entries: dict[str, SourceEntry] = {e.id: e for e in entries}
        self._sha256 = file_sha256

    # -- Public API -----------------------------------------------------------

    def __getitem__(self, key: str) -> SourceEntry:
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(
                f"SRC['{key}'] not found in SOURCES.yaml. "
                f"Register it before using it in code."
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def all_ids(self) -> list[str]:
        return sorted(self._entries.keys())

    def placeholders(self) -> list[str]:
        """IDs tagged phase2_replace=true — values not yet data-fitted."""
        return [e.id for e in self._entries.values() if e.phase2_replace]

    @property
    def sha256(self) -> str:
        return self._sha256

    def to_dict(self) -> dict:
        """Serialise to plain dict (useful for /api/v1/sources endpoint)."""
        return {k: {
            "value": v.value,
            "unit": v.unit,
            "source_url": v.source_url,
            "source_type": v.source_type,
            "retrieved_at": v.retrieved_at,
            "license": v.license,
            "note": v.note,
            "phase2_replace": v.phase2_replace,
        } for k, v in self._entries.items()}


# -- Loader -------------------------------------------------------------------

def _load_yaml_simple(path: pathlib.Path) -> list[dict]:
    """
    Minimal YAML list-of-dicts parser — no pyyaml dependency required for the
    simple subset used in SOURCES.yaml.  Falls back to `yaml` if available.

    The file must be a YAML sequence of mappings (list of dicts).
    """
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, list):
            raise ValueError("SOURCES.yaml must be a YAML sequence (list) at the top level.")
        return data
    except ImportError:
        pass

    # Fallback: manual parser for the strict subset this file uses.
    return _manual_yaml_parse(path)


def _manual_yaml_parse(path: pathlib.Path) -> list[dict]:
    """
    Hand-rolled parser for the YAML subset in SOURCES.yaml:
      - Sequence of block mappings, each entry starting with `- id: ...`
      - Scalar and simple list values only
      - No nested dicts beyond `value:` lists
    """
    raw = path.read_text(encoding="utf-8")
    entries: list[dict] = []
    current: dict | None = None

    in_list_value = False
    list_key = ""
    list_values: list[str] = []

    for raw_line in raw.splitlines():
        line = raw_line.rstrip()

        # Skip comments and blank lines
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            if in_list_value and current is not None:
                # blank line may end a list block
                pass
            continue

        indent = len(line) - len(stripped)

        # New entry
        if stripped.startswith("- id:"):
            if current is not None:
                if in_list_value:
                    current[list_key] = list_values[:]
                    in_list_value = False
                entries.append(current)
            current = {"id": stripped[len("- id:"):].strip().strip("'\"")}
            in_list_value = False
            continue

        # Continuation list item (indent 4 with "- " prefix)
        if in_list_value and stripped.startswith("- ") and indent >= 4:
            list_values.append(stripped[2:].strip().strip("'\""))
            continue

        if in_list_value:
            # End of list block
            if current is not None:
                current[list_key] = list_values[:]
            in_list_value = False

        if current is None:
            continue

        # Key: value pair (indent 2)
        if ":" in stripped and indent == 2:
            key, _, rest = stripped.partition(":")
            key = key.strip()
            val = rest.strip().strip("'\"")
            if val == "":
                # multi-line list follows
                in_list_value = True
                list_key = key
                list_values = []
            elif val.lower() == "true":
                current[key] = True
            elif val.lower() == "false":
                current[key] = False
            else:
                try:
                    current[key] = float(val) if "." in val else int(val)
                except ValueError:
                    current[key] = val

    if current is not None:
        if in_list_value:
            current[list_key] = list_values[:]
        entries.append(current)

    return entries


def _build_registry(path: pathlib.Path) -> SourcesRegistry:
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    data = _load_yaml_simple(path)
    entries: list[SourceEntry] = []
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            entry = SourceEntry(
                id=str(item["id"]),
                value=item.get("value"),
                unit=str(item.get("unit", "")),
                source_url=str(item.get("source_url", "")),
                source_type=str(item.get("source_type", "empirical")),
                retrieved_at=str(item.get("retrieved_at", "")),
                license=str(item.get("license", "")),
                note=str(item.get("note", "")),
                phase2_replace=bool(item.get("phase2_replace", False)),
            )
            entries.append(entry)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"SOURCES.yaml parse error for entry id='{item.get('id','?')}': {exc}") from exc

    return SourcesRegistry(entries, sha256)


def get_registry() -> SourcesRegistry:
    """Return the module-level singleton SourcesRegistry (thread-safe)."""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = _build_registry(_SOURCES_PATH)
    return _INSTANCE


def reload_registry() -> SourcesRegistry:
    """Force reload from disk — use in tests or after file edits."""
    global _INSTANCE
    with _LOCK:
        _INSTANCE = _build_registry(_SOURCES_PATH)
    return _INSTANCE

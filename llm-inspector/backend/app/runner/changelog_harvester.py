"""
runner/changelog_harvester.py — v17 Phase 7: Tier-3 documentation harvesting.

The ``/v1/models`` endpoints picked up by Phase 6 do *not* expose the
fields the inspector needs most for cutoff/feature reasoning:
``cutoff_date``, ``supports_thinking``, ``deprecation_date``,
human-readable pricing, etc.  Those live in vendor blog posts and docs
HTML, which are inconsistent in shape and frequently restructured.

Strategy (UPGRADE_PLAN_V17.md §10):

  1. Pull a small set of RSS / Atom feeds + docs HTML pages
  2. Cheaply slice each fetched document into a small text envelope
     (no BeautifulSoup dependency — stdlib ``html.parser``)
  3. Hand the slice to an LLM (``JUDGE_API_URL``) with a strict-JSON
     extraction prompt
  4. Validate the extracted JSON against ``ExtractedRecord``; *anti-
     hallucination* gate rejects records whose ``evidence_quote`` does
     not appear verbatim in the source text
  5. Upsert survivors via :mod:`app.repository.registry_repo` with
     ``data_source='changelog'`` and ``confidence=0.85`` so they cannot
     overwrite higher-priority sources

This module ships:
  * Source registry (``CHANGELOG_SOURCES``)
  * Pure helpers (``strip_html``, ``validate_extracted_record``,
    ``apply_extracted_records``)
  * ``run_harvest()`` orchestrator
  * CLI: ``python -m app.runner.changelog_harvester --once``

LLM call is delegated to a pluggable callable so the offline tests can
inject canned extractions.
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable

from app.core.logging import get_logger
from app.repository import registry_repo

logger = get_logger(__name__)


# ── Source registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChangelogSource:
    name: str
    url: str
    vendor: str
    kind: str           # "rss" | "atom" | "html"


CHANGELOG_SOURCES: tuple[ChangelogSource, ...] = (
    ChangelogSource("openai_blog",      "https://openai.com/blog/rss/",                              "openai",    "rss"),
    ChangelogSource("openai_models",    "https://platform.openai.com/docs/models",                   "openai",    "html"),
    ChangelogSource("anthropic_news",   "https://www.anthropic.com/news",                            "anthropic", "html"),
    ChangelogSource("anthropic_models", "https://docs.anthropic.com/en/docs/about-claude/models",    "anthropic", "html"),
    ChangelogSource("google_blog",      "https://blog.google/technology/google-deepmind/rss/",       "google",    "rss"),
    ChangelogSource("xai_news",         "https://x.ai/news",                                         "xai",       "html"),
)


# ── HTTP + HTML helpers ─────────────────────────────────────────────────────


def _build_ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import]
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CTX = _build_ssl_ctx()


def fetch_text(url: str, timeout: int = 20) -> str | None:
    """GET ``url`` and return decoded body, or None on any failure."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html, application/rss+xml, application/atom+xml, */*",
            "User-Agent": "LLMInspector/17.0 ChangelogHarvester",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
            data = resp.read()
            charset = "utf-8"
            try:
                ctype = resp.headers.get("Content-Type") or ""
                if "charset=" in ctype:
                    charset = ctype.split("charset=", 1)[1].strip().split(";")[0]
            except Exception:
                pass
            return data.decode(charset, errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        logger.warning("changelog_harvester: fetch failed", url=url, error=str(getattr(e, "reason", e)))
        return None
    except Exception as e:
        logger.warning("changelog_harvester: fetch error", url=url, error=str(e)[:200])
        return None


class _Stripper(HTMLParser):
    """Minimal HTML→text stripper. Drops <script>/<style> contents."""
    _SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):  # noqa: D401
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self._buf.append(data)

    def get_text(self) -> str:
        return "".join(self._buf)


_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(html: str, *, max_chars: int = 8000) -> str:
    """Convert HTML / RSS body to whitespace-collapsed plain text."""
    if not html:
        return ""
    parser = _Stripper()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # On malformed input, fall back to coarse regex strip.
        text = re.sub(r"<[^>]+>", " ", html)
    else:
        text = parser.get_text()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " …"
    return text


# ── Extraction record + validation ──────────────────────────────────────────


@dataclass
class ExtractedRecord:
    model_id: str
    release_date: str | None = None        # ISO date
    cutoff_date: str | None = None
    context_window: int | None = None
    supports_thinking: bool | None = None
    deprecated: bool | None = None
    deprecation_date: str | None = None
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None
    evidence_quote: str = ""

    def to_registry_record(self, vendor: str, source_url: str) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "model_id": self.model_id,
            "vendor": vendor,
            "data_source": "changelog",
            "confidence": 0.85,
            "raw_metadata": {
                "source_url": source_url,
                "release_date": self.release_date,
                "evidence_quote": self.evidence_quote[:280],
            },
        }
        if self.cutoff_date is not None:
            rec["cutoff_date"] = self.cutoff_date
        if self.context_window is not None:
            rec["context_window"] = int(self.context_window)
        if self.supports_thinking is not None:
            rec["supports_thinking"] = bool(self.supports_thinking)
        if self.input_price_per_mtok is not None:
            rec["input_price_usd"] = float(self.input_price_per_mtok)
        if self.output_price_per_mtok is not None:
            rec["output_price_usd"] = float(self.output_price_per_mtok)
        return rec


_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._\-:/]{2,80}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _norm(text: str) -> str:
    """Whitespace-collapsed lower-case."""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def validate_extracted_record(
    item: dict[str, Any], source_text: str
) -> tuple[ExtractedRecord | None, str]:
    """Validate one LLM-emitted dict.  Returns (record, reason_if_rejected)."""
    if not isinstance(item, dict):
        return None, "not_a_dict"

    mid = item.get("model_id")
    if not isinstance(mid, str) or not _MODEL_ID_RE.match(mid):
        return None, "invalid_model_id"

    quote = item.get("evidence_quote")
    if not isinstance(quote, str) or len(quote) < 10:
        return None, "evidence_quote_missing_or_short"

    # Anti-hallucination gate: the quote must appear (case-insensitively,
    # whitespace-normalised) in the source.  This catches the most common
    # LLM failure mode where the model invents plausible-sounding but
    # absent justifications.
    if _norm(quote)[:160] not in _norm(source_text):
        return None, "evidence_quote_not_in_source"

    # Date format checks (lenient — None passes through)
    for date_field in ("release_date", "cutoff_date", "deprecation_date"):
        v = item.get(date_field)
        if v is not None:
            if not isinstance(v, str) or not _DATE_RE.match(v):
                return None, f"invalid_{date_field}"

    # Numeric coercion
    cw = item.get("context_window")
    if cw is not None:
        try:
            cw_int = int(cw)
            if cw_int < 0 or cw_int > 100_000_000:
                return None, "context_window_out_of_range"
        except (TypeError, ValueError):
            return None, "context_window_not_int"
    else:
        cw_int = None

    def _coerce_price(name: str) -> tuple[float | None, str | None]:
        v = item.get(name)
        if v is None:
            return None, None
        try:
            f = float(v)
            if f < 0 or f > 10000:
                return None, f"{name}_out_of_range"
            return f, None
        except (TypeError, ValueError):
            return None, f"{name}_not_numeric"

    in_price, err = _coerce_price("input_price_per_mtok")
    if err:
        return None, err
    out_price, err = _coerce_price("output_price_per_mtok")
    if err:
        return None, err

    rec = ExtractedRecord(
        model_id=mid,
        release_date=item.get("release_date"),
        cutoff_date=item.get("cutoff_date"),
        context_window=cw_int,
        supports_thinking=(bool(item["supports_thinking"])
                           if item.get("supports_thinking") is not None else None),
        deprecated=(bool(item["deprecated"])
                    if item.get("deprecated") is not None else None),
        deprecation_date=item.get("deprecation_date"),
        input_price_per_mtok=in_price,
        output_price_per_mtok=out_price,
        evidence_quote=quote,
    )
    return rec, ""


# ── LLM extraction prompt + pluggable transport ────────────────────────────


EXTRACTION_PROMPT_TEMPLATE = """You are a model-metadata extractor. Given the
text from a vendor blog or docs page, return a STRICT JSON list of objects,
one per model release/update mentioned. Each object MUST have this schema:

{{
  "model_id": "<canonical-id-as-used-by-the-public-api>",
  "release_date": "YYYY-MM-DD" | null,
  "cutoff_date": "YYYY-MM-DD" | null,
  "context_window": <integer> | null,
  "supports_thinking": <bool> | null,
  "deprecated": <bool> | null,
  "deprecation_date": "YYYY-MM-DD" | null,
  "input_price_per_mtok": <number, USD> | null,
  "output_price_per_mtok": <number, USD> | null,
  "evidence_quote": "<= 200 chars copied verbatim from the input"
}}

Rules:
  - Only include items explicitly stated in the text.
  - Never guess or compute values.
  - The evidence_quote must be a verbatim substring of the input text.
  - Prices must be USD per 1 million tokens (USD/Mtok).
  - Output ONLY the JSON list, no markdown fences, no commentary.

INPUT TEXT
==========
{text}
==========
"""


LLMExtractor = Callable[[str], list[dict[str, Any]]]


def _default_llm_extractor(_text: str) -> list[dict[str, Any]]:
    """Default: no LLM available → return empty list with a warning.

    Phase 7 production deploys must inject a real callable wired to
    ``JUDGE_API_URL`` (see ``UPGRADE_PLAN_V17.md`` §10.2).  We do not
    hardcode that wiring here so the harvester remains testable offline
    and decoupled from the judge layer.
    """
    logger.warning(
        "changelog_harvester: no LLM extractor configured; "
        "wire one via run_harvest(extractor=...) to enable Phase 7"
    )
    return []


# ── Orchestration ───────────────────────────────────────────────────────────


@dataclass
class HarvestResult:
    source_name: str
    fetched: bool = False
    text_chars: int = 0
    raw_records: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    upserted_model_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "fetched": self.fetched,
            "text_chars": self.text_chars,
            "raw_records": self.raw_records,
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "rejection_reasons": self.rejection_reasons,
            "upserted_model_ids": self.upserted_model_ids,
        }


def apply_extracted_records(
    raw_items: list[dict[str, Any]],
    source_text: str,
    vendor: str,
    source_url: str,
) -> HarvestResult:
    """Validate and upsert a list of LLM-emitted records for one source."""
    out = HarvestResult(source_name=source_url)
    out.raw_records = len(raw_items)
    for item in raw_items:
        rec, reason = validate_extracted_record(item, source_text)
        if rec is None:
            out.rejected_records += 1
            out.rejection_reasons[reason] = out.rejection_reasons.get(reason, 0) + 1
            continue
        try:
            registry_repo.upsert_model(rec.to_registry_record(vendor, source_url))
            out.accepted_records += 1
            out.upserted_model_ids.append(rec.model_id)
        except Exception as e:
            out.rejected_records += 1
            out.rejection_reasons["upsert_failed"] = (
                out.rejection_reasons.get("upsert_failed", 0) + 1
            )
            logger.warning("upsert failed for changelog record",
                           model_id=rec.model_id, error=str(e))
    return out


def run_harvest(
    sources: tuple[ChangelogSource, ...] = CHANGELOG_SOURCES,
    extractor: LLMExtractor | None = None,
    fetcher: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Run the full harvest pipeline against ``sources``.

    Both ``extractor`` and ``fetcher`` are injectable for testing.
    """
    extract: LLMExtractor = extractor or _default_llm_extractor
    fetch: Callable[[str], str | None] = fetcher or fetch_text

    started = int(time.time())
    per_source: list[HarvestResult] = []
    for src in sources:
        html = fetch(src.url)
        if not html:
            per_source.append(HarvestResult(source_name=src.name, fetched=False))
            continue
        text = strip_html(html)
        if not text:
            per_source.append(HarvestResult(source_name=src.name, fetched=True, text_chars=0))
            continue
        try:
            raw_items = extract(EXTRACTION_PROMPT_TEMPLATE.format(text=text)) or []
        except Exception as e:
            logger.warning("changelog_harvester: extractor raised", source=src.name, error=str(e))
            raw_items = []
        result = apply_extracted_records(raw_items, text, src.vendor, src.url)
        result.source_name = src.name
        result.fetched = True
        result.text_chars = len(text)
        per_source.append(result)

    finished = int(time.time())
    return {
        "started_at": started,
        "finished_at": finished,
        "duration_sec": finished - started,
        "per_source": [r.to_dict() for r in per_source],
        "total_accepted": sum(r.accepted_records for r in per_source),
        "total_rejected": sum(r.rejected_records for r in per_source),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true",
                        help="Run one harvest cycle then exit (default).")
    args = parser.parse_args()
    _ = args  # currently single-shot only

    from app.core.db import init_db, get_conn
    from app.core.db_migrations import migrate
    init_db()
    migrate(get_conn())

    report = run_harvest()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "ChangelogSource",
    "CHANGELOG_SOURCES",
    "ExtractedRecord",
    "HarvestResult",
    "EXTRACTION_PROMPT_TEMPLATE",
    "fetch_text",
    "strip_html",
    "validate_extracted_record",
    "apply_extracted_records",
    "run_harvest",
]

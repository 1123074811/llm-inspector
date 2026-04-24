"""
runner/cache_strategy.py - v15 Phase 10: Token Efficiency via Cache Strategy.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from app.core.db import get_conn, now_iso
from app.core.logging import get_logger
from app.core.schemas import LLMResponse
logger = get_logger(__name__)

@dataclass
class CacheMetricsSnapshot:
    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    tokens_spent_on_misses: int = 0
    hit_rate: float = 0.0
    estimated_cost_savings: float = 0.0
    cache_size: int = 0
    expired_entries: int = 0
    warming_requests: int = 0

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "tokens_saved": self.tokens_saved,
            "tokens_spent_on_misses": self.tokens_spent_on_misses,
            "hit_rate": round(self.hit_rate, 4),
            "estimated_cost_savings": round(self.estimated_cost_savings, 6),
            "cache_size": self.cache_size,
            "expired_entries": self.expired_entries,
            "warming_requests": self.warming_requests,
        }

@dataclass
class TTLPolicy:
    default_hours: int = 24
    long_hours: int = 72
    short_hours: int = 6
    no_cache: bool = False

class CacheStrategy:
    CATEGORY_TTL: dict[str, TTLPolicy] = {
        "reasoning": TTLPolicy(default_hours=24),
        "knowledge": TTLPolicy(short_hours=6),
        "instruction": TTLPolicy(default_hours=24),
        "coding": TTLPolicy(default_hours=24),
        "safety": TTLPolicy(default_hours=48),
        "extraction": TTLPolicy(long_hours=72),
        "identity": TTLPolicy(long_hours=72),
        "tokenizer": TTLPolicy(long_hours=168),
        "timing": TTLPolicy(short_hours=4),
        "multilingual": TTLPolicy(default_hours=24),
        "adversarial": TTLPolicy(short_hours=12),
    }
    DEFAULT_TTL = TTLPolicy()

    def __init__(self) -> None:
        self._lock = Lock()
        self._total_requests: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._tokens_saved: int = 0
        self._tokens_spent_on_misses: int = 0
        self._warming_count: int = 0

    def get(self, cache_key: str) -> LLMResponse | None:
        with self._lock:
            self._total_requests += 1
        try:
            conn = get_conn()
            row = conn.execute(
                "SELECT response_json FROM llm_response_cache WHERE cache_key=? AND expires_at > ?",
                (cache_key, now_iso())
            ).fetchone()
            if row:
                data = json.loads(row["response_json"])
                resp = LLMResponse(**data)
                with self._lock:
                    self._hits += 1
                    self._tokens_saved += (resp.usage_total_tokens or 0)
                return resp
        except Exception as e:
            logger.warning("Cache get error", error=str(e))
        with self._lock:
            self._misses += 1
        return None

    def set(self, cache_key: str, resp: LLMResponse, category: str = "") -> None:
        if not resp.ok or resp.error_type:
            return
        ttl = self._resolve_ttl(category)
        if ttl.no_cache:
            return
        try:
            expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl.default_hours)).isoformat()
            conn = get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO llm_response_cache (cache_key, response_json, created_at, expires_at) VALUES (?,?,?,?)",
                (cache_key, json.dumps(resp.to_dict()), now_iso(), expires_at)
            )
            conn.commit()
        except Exception as e:
            logger.warning("Cache set error", error=str(e))

    def build_key(self, base_url: str, payload: dict) -> str:
        payload_str = json.dumps(payload, sort_keys=True)
        key_src = base_url + ":" + payload_str
        return hashlib.sha256(key_src.encode()).hexdigest()

    def warm(self, base_url: str, payload: dict, response: LLMResponse, category: str = "") -> None:
        self.set(self.build_key(base_url, payload), response, category=category)
        with self._lock:
            self._warming_count += 1

    def invalidate(self, cache_key: str) -> bool:
        try:
            conn = get_conn()
            conn.execute("DELETE FROM llm_response_cache WHERE cache_key=?", (cache_key,))
            conn.commit()
            return True
        except Exception as e:
            logger.warning("Cache invalidate error", error=str(e))
            return False

    def evict_expired(self) -> int:
        try:
            conn = get_conn()
            cursor = conn.execute("DELETE FROM llm_response_cache WHERE expires_at <= ?", (now_iso(),))
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            logger.warning("Cache eviction error", error=str(e))
            return 0

    def cache_size(self) -> int:
        try:
            conn = get_conn()
            row = conn.execute("SELECT COUNT(*) AS cnt FROM llm_response_cache WHERE expires_at > ?", (now_iso(),)).fetchone()
            return row["cnt"] if row else 0
        except Exception as e:
            logger.warning("Cache size error", error=str(e))
            return 0

    def expired_count(self) -> int:
        try:
            conn = get_conn()
            row = conn.execute("SELECT COUNT(*) AS cnt FROM llm_response_cache WHERE expires_at <= ?", (now_iso(),)).fetchone()
            return row["cnt"] if row else 0
        except Exception as e:
            logger.warning("Cache expired count error", error=str(e))
            return 0

    def snapshot(self) -> CacheMetricsSnapshot:
        with self._lock:
            total = self._total_requests
            hits = self._hits
            misses = self._misses
            tokens_saved = self._tokens_saved
            tokens_miss = self._tokens_spent_on_misses
            warming = self._warming_count
        size = self.cache_size()
        expired = self.expired_count()
        hit_rate = hits / total if total > 0 else 0.0
        cost_savings = tokens_saved * 0.00001
        return CacheMetricsSnapshot(
            total_requests=total, hits=hits, misses=misses,
            tokens_saved=tokens_saved, tokens_spent_on_misses=tokens_miss,
            hit_rate=hit_rate, estimated_cost_savings=cost_savings,
            cache_size=size, expired_entries=expired, warming_requests=warming,
        )

    def record_miss_tokens(self, tokens: int) -> None:
        with self._lock:
            self._tokens_spent_on_misses += tokens

    def reset_metrics(self) -> None:
        with self._lock:
            self._total_requests = 0
            self._hits = 0
            self._misses = 0
            self._tokens_saved = 0
            self._tokens_spent_on_misses = 0
            self._warming_count = 0

    def _resolve_ttl(self, category: str) -> TTLPolicy:
        if not category:
            return self.DEFAULT_TTL
        return self.CATEGORY_TTL.get(category, self.DEFAULT_TTL)

cache_strategy = CacheStrategy()

def get_cache_strategy() -> CacheStrategy:
    return cache_strategy

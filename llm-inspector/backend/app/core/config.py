"""
Configuration — reads from environment variables or .env file.
No external dependencies.
"""
import os
import pathlib

def _load_env_file():
    """Load .env file if present."""
    env_path = pathlib.Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env_file()


class Settings:
    # App
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    STRICT_PROVENANCE: bool = os.getenv("STRICT_PROVENANCE", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

    # Database (SQLite for portability; swap DATABASE_URL for PostgreSQL)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./llm_inspector.db"
    )

    # Security
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")  # 32-byte base64; generated if missing
    API_KEY_TTL_HOURS: int = int(os.getenv("API_KEY_TTL_HOURS", "72"))

    # Request behaviour
    DEFAULT_REQUEST_TIMEOUT_SEC: int = int(os.getenv("DEFAULT_REQUEST_TIMEOUT_SEC", "60"))
    MAX_STREAM_CHUNKS: int = int(os.getenv("MAX_STREAM_CHUNKS", "512"))
    INTER_REQUEST_DELAY_MS: int = int(os.getenv("INTER_REQUEST_DELAY_MS", "150"))

    # Data retention
    RAW_RESPONSE_TTL_DAYS: int = int(os.getenv("RAW_RESPONSE_TTL_DAYS", "7"))
    STREAM_CHUNKS_TTL_DAYS: int = int(os.getenv("STREAM_CHUNKS_TTL_DAYS", "3"))

    # Task backend (in-process thread pool when Celery unavailable)
    USE_CELERY: bool = os.getenv("USE_CELERY", "false").lower() == "true"
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Predetection
    PREDETECT_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("PREDETECT_CONFIDENCE_THRESHOLD", "0.85")
    )

    # Adaptive suite controls (token-cost vs confidence)
    SENTINEL_SIZE: int = int(os.getenv("SENTINEL_SIZE", "10"))
    CORE_SIZE: int = int(os.getenv("CORE_SIZE", "12"))
    EXPANSION_SIZE: int = int(os.getenv("EXPANSION_SIZE", "10"))
    ARBITRATION_MAX: int = int(os.getenv("ARBITRATION_MAX", "6"))
    DEFAULT_MAX_TOKENS_CAP: int = int(os.getenv("DEFAULT_MAX_TOKENS_CAP", "600"))
    LONG_FORM_MAX_TOKENS_CAP: int = int(os.getenv("LONG_FORM_MAX_TOKENS_CAP", "1200"))

    # Token budget guards (per-run token spending limits)
    TOKEN_BUDGET_QUICK: int = int(os.getenv("TOKEN_BUDGET_QUICK", "15000"))
    TOKEN_BUDGET_STANDARD: int = int(os.getenv("TOKEN_BUDGET_STANDARD", "40000"))
    TOKEN_BUDGET_DEEP: int = int(os.getenv("TOKEN_BUDGET_DEEP", "100000"))
    TOKEN_BUDGET_FULL: int = int(os.getenv("TOKEN_BUDGET_FULL", "100000"))  # legacy alias

    # Semantic Judge (LLM-as-Judge for open-ended evaluation)
    JUDGE_API_URL: str = os.getenv("JUDGE_API_URL", "")
    JUDGE_API_KEY: str = os.getenv("JUDGE_API_KEY", "")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    JUDGE_TIMEOUT: int = int(os.getenv("JUDGE_TIMEOUT", "15"))

    # Theta / relative scale
    THETA_METHOD: str = os.getenv("THETA_METHOD", "rasch_1pl")
    THETA_BOOTSTRAP_B: int = int(os.getenv("THETA_BOOTSTRAP_B", "200"))
    THETA_BOOTSTRAP_B_MIN: int = int(os.getenv("THETA_BOOTSTRAP_B_MIN", "50"))
    THETA_CI_STOP_WIDTH: float = float(os.getenv("THETA_CI_STOP_WIDTH", "0.25"))
    THETA_DELTA_STOP: float = float(os.getenv("THETA_DELTA_STOP", "0.45"))
    THETA_SCALE_FOR_WIN_PROB: float = float(os.getenv("THETA_SCALE_FOR_WIN_PROB", "0.6"))
    CALIBRATION_VERSION: str = os.getenv("CALIBRATION_VERSION", "v1")
    SUITE_VERSION: str = os.getenv("SUITE_VERSION", "v10")

    # Golden baseline compare thresholds (display-score units)
    BASELINE_MATCH_COSINE_THRESHOLD: float = float(
        os.getenv("BASELINE_MATCH_COSINE_THRESHOLD", "0.95")
    )
    BASELINE_MATCH_SCORE_DELTA_MAX: int = int(
        os.getenv("BASELINE_MATCH_SCORE_DELTA_MAX", "500")
    )

    # Production hardening
    MAX_CONCURRENT_RUNS: int = int(os.getenv("MAX_CONCURRENT_RUNS", "5"))
    MAX_REQUEST_RETRIES: int = int(os.getenv("MAX_REQUEST_RETRIES", "3"))
    RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "60"))

    # Phase C: execution concurrency tuning
    CONCURRENCY_QUICK: int = int(os.getenv("CONCURRENCY_QUICK", "12"))
    CONCURRENCY_STANDARD: int = int(os.getenv("CONCURRENCY_STANDARD", "8"))
    CONCURRENCY_DEEP: int = int(os.getenv("CONCURRENCY_DEEP", "3"))

    # Phase C: benchmark cache TTL (seconds)
    BENCHMARK_CACHE_TTL_SEC: int = int(os.getenv("BENCHMARK_CACHE_TTL_SEC", "120"))

    # Phase D: async pipeline rollout controls
    ASYNC_PIPELINE_ENABLED: bool = os.getenv("ASYNC_PIPELINE_ENABLED", "false").lower() == "true"
    ASYNC_PIPELINE_FALLBACK_SYNC: bool = os.getenv("ASYNC_PIPELINE_FALLBACK_SYNC", "true").lower() == "true"

    # Verdict engine overrides (JSON string of threshold overrides)
    VERDICT_TRUSTED_THRESHOLD: int = int(os.getenv("VERDICT_TRUSTED_THRESHOLD", "80"))
    VERDICT_SUSPICIOUS_THRESHOLD: int = int(os.getenv("VERDICT_SUSPICIOUS_THRESHOLD", "60"))
    VERDICT_HIGH_RISK_THRESHOLD: int = int(os.getenv("VERDICT_HIGH_RISK_THRESHOLD", "40"))

    # v12 Phase 2: Semantic Judge V3 and Hallucination Detector V3
    USE_EXTERNAL_LLM: bool = os.getenv("USE_EXTERNAL_LLM", "false").lower() == "true"
    USE_KNOWLEDGE_GRAPH: bool = os.getenv("USE_KNOWLEDGE_GRAPH", "false").lower() == "true"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() in ("production", "prod")

    def _ensure_encryption_key(self) -> bytes:
        """Return 32-byte AES key, auto-generating one if not set."""
        import base64, secrets
        if self.ENCRYPTION_KEY:
            raw = base64.b64decode(self.ENCRYPTION_KEY)
            assert len(raw) == 32, "ENCRYPTION_KEY must decode to exactly 32 bytes"
            return raw
        # Dev fallback: generate deterministic key from hostname (NOT for production)
        import hashlib, socket
        seed = socket.gethostname().encode() + b"llm-inspector-dev-key"
        return hashlib.sha256(seed).digest()

    @property
    def aes_key(self) -> bytes:
        return self._ensure_encryption_key()


settings = Settings()

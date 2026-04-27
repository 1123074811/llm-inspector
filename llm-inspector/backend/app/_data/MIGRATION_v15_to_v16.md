# Migration Guide: v15 → v16

## Database Schema Changes

v16 adds 5 new columns. Run the migration script:

```bash
python -c "from app.core.db_migrations import migrate_v15_to_v16; migrate_v15_to_v16()"
```

### New Columns

| Table | Column | Type | Default | Description |
|-------|--------|------|---------|-------------|
| `runs` | `finish_reason` | TEXT | NULL | Upstream finish_reason |
| `runs` | `http_status` | INTEGER | NULL | Last HTTP status from upstream |
| `runs` | `excluded_count` | INTEGER | 0 | Samples excluded from scoring |
| `runs` | `eligible_total` | INTEGER | 0 | Eligible samples after exclusion |
| `runs` | `last_heartbeat_at` | TEXT | NULL | ISO timestamp of last SSE heartbeat |

## API Changes

### New Endpoints

| Method | Path | Phase | Description |
|--------|------|-------|-------------|
| GET | `/api/v1/runs/{id}/token-audit` | 7 | Token consumption audit |
| GET | `/api/v1/runs/{id}/real-model-card` | 9 | Real Model Card data |

### New SSE Event Types (Phase 8)

`retry.truncation`, `retry.5xx`, `retry.decode`, `sample.excluded`,
`identity.exposure_detected`, `system_prompt.leaked`, `model_list.probed`, `judge_chain.step`

## Configuration Changes

New env vars: `PREFLIGHT_TIMEOUT_S`, `OFFICIAL_ENDPOINT_ENABLED`,
`RETRY_MAX_5XX`, `RETRY_MAX_TRUNCATION`, `RETRY_MAX_DECODE`

## Breaking Changes

- `ErrorCode.E_MODEL_NOT_FOUND` duplicate removed
- `RetryConfig` now includes `jitter_ratio` (default 0.1)
- `VerificationResult` has new fields `kg_conflict` and `degraded`
- `ScoreCard.to_dict()` includes `coverage`, `weight_provenance_trace`, `weighted_ece`, `excluded_case_count`

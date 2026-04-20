# Migration Guide: v13 → v14

## Overview

v14 introduces major improvements to the judge system, scoring pipeline, predetect layers, and
frontend. This guide covers breaking changes and required adaptations.

---

## API Changes

### New Endpoints (v14)

| Endpoint | Phase | Description |
|---|---|---|
| `GET /api/v14/runs/{id}/token-analysis` | Phase 6 | Token usage and optimizer stats |
| `GET /api/v14/runs/{id}/identity-exposure` | Phase 3 | Bayesian identity collision result |
| `GET /api/v14/runs/{id}/predetect-trace` | Phase 5 | Per-layer predetect timing trace |
| `GET /api/v1/circuit-breaker/history` | Phase 7 | Circuit breaker event history |

### Changed Endpoints

- `GET /api/v1/leaderboard` now accepts `limit` and `offset` query parameters for pagination.
  - Default `limit=20`, `offset=0`.
  - Example: `/api/v1/leaderboard?limit=20&offset=40`

### Deprecated / Removed

- `benchmark_profiles` table is dropped in Migration003; any direct queries against it will fail.
- The `full` and `extraction` test mode aliases still work (auto-mapped to `deep`) but are
  considered legacy. Use `deep` directly.

---

## ScoreCard Changes

### Fields that now return `null`

In v13, several sub-scores defaulted to `50.0` when no data was available. In v14 these return
`null` (Python `None` → JSON `null`) to clearly distinguish "not measured" from "scored 50".

Affected fields in `scorecard.breakdown`:
- `reasoning` — `null` when no reasoning cases ran
- `adversarial_reasoning` — `null` when no adversarial cases ran
- `coding` — `null` when no coding cases ran
- `speed` — `null` when no timing data collected
- `stability` — `null` when < 2 samples available

Top-level:
- `total_score` — `null` if no cases completed at all (was `0.0`)

### New Fields (v14)

```json
{
  "v13": {
    "completeness": 0.875
  },
  "token_analysis": {
    "prompt_optimizer_used": true,
    "tokens_saved_estimate": 3200,
    "counting_method": "tiktoken-cl100k"
  },
  "skipped_cases": ["case_001", "case_002"]
}
```

- `v13.completeness` — fraction of capability dimensions with non-null data (0.0–1.0)
- `token_analysis` — token counting and optimizer stats (Phase 6)
- `skipped_cases` — list of case IDs that were intentionally skipped (Phase 7)

### v13 Backward-Compatible Block

The `scorecard.v13` block from v13 is unchanged and still present.

---

## Frontend Adaptation

### Null score display

v13 frontend rendered null scores as `0` or `50`. v14 should render them as `N/A`:

```javascript
// Before (v13)
element.textContent = score.toFixed(1);         // crashes on null
element.textContent = value || '0';              // shows '0' for null

// After (v14) — use fmtScore()
element.innerHTML = fmtScore(score);             // returns '—' span for null
// or for plain text contexts:
function fmtScorePlain(v, digits=1) {
    if (v === null || v === undefined) return 'N/A';
    return typeof v === 'number' ? v.toFixed(digits) : String(v);
}
```

### safeFetch wrapper

All direct `fetch()` calls in the frontend should be replaced with `safeFetch()` which provides
automatic error toasts and null-safe returns:

```javascript
// Before
const resp = await fetch('/api/v1/runs/' + id + '/report');
const data = await resp.json();

// After
const resp = await safeFetch('/api/v1/runs/' + id + '/report');
if (!resp) return;  // error toast already shown
const data = await resp.json().catch(() => null);
```

### Radar chart: filter null dimensions

In v14, some scorecard dimensions may be `null`. The radar chart renderer now filters these out
before building the `indicator` array. If fewer than 5 dimensions have data, it falls back to a
bar chart. No frontend action required if using the updated `renderRadarChart()`.

### Leaderboard pagination

The leaderboard now uses server-side pagination. `loadLeaderboard(offset)` accepts an offset
parameter. The client-side search filters the current page only.

---

## Judge Method Changes

### New Judge Methods (v14 Phase 4)

| Method | Description | Reference |
|---|---|---|
| `numeric_tolerance` | Relative error ≤ 5% (configurable) | NIST SP 330-2019 |
| `multi_choice_verified` | Strict letter-match for MMLU-style | Hendrycks et al. 2021 |
| `semantic_entailment` | NLI entailment check, 3-tier fallback | Reimers & Gurevych 2019 |

### hallucination_v2

In v14 Phase 4, `hallucination_v2` performs real DBpedia SPARQL queries to cross-check claims
against a live knowledge graph. This requires network access. If DBpedia is unreachable, it
falls back to heuristic-only detection.

**Impact**: fact-checking latency increases by ~200–800 ms per case using this judge. Disable
with `HALLUCINATION_KG_ENABLED=false` if network is unavailable.

### JudgeChainRunner (transparent_judge.py)

v14 adds a 4-tier degradation chain:
1. External LLM judge (requires `JUDGE_API_URL`)
2. Local NLI entailment
3. Rule-based judge
4. Hallucination detection fallback

Fleiss's κ (≥3 judges) replaces Cohen's κ when more than 2 judge instances are active.

---

## Database Migration

Migrations run automatically on server start via `db_migrations.migrate()`.

| Migration | Description |
|---|---|
| Migration001 | Adds `evaluation_mode`, `calibration_case_id` columns to `test_runs` |
| Migration002 | Adds `scoring_profile_version`, `calibration_tag` columns |
| Migration003 | Drops `benchmark_profiles` table (replaced by `golden_baselines`) |
| Migration004 | Adds `identity_exposure_result TEXT` column to `test_runs` |

### Manual migration (if auto-migrate disabled)

```python
from app.core.db_migrations import migrate
migrate()
```

---

## Configuration Changes

### New environment variables (v14)

```env
# Phase 5: Timing side-channel
TIMING_SIDE_CHANNEL_ENABLED=true   # Enable L18 TTFT/TPS analysis

# Phase 6: Token optimizer
TOKEN_OPTIMIZER_ENABLED=true       # Enable dynamic few-shot selection
TOKEN_BUDGET_QUICK=15000           # Unchanged
TOKEN_BUDGET_STANDARD=40000        # Unchanged
TOKEN_BUDGET_DEEP=100000           # Unchanged

# Phase 4: Hallucination KG
HALLUCINATION_KG_ENABLED=true      # Real DBpedia queries (requires network)
```

---

## PreDetect Pipeline

v14 extends the predetect pipeline to 20 layers (was 16 in v13):

| Layer | Name | Added |
|---|---|---|
| L17 | Identity Exposure Engine | v14 Phase 3 |
| L18 | Response Timing Side-Channel | v14 Phase 5 |
| L19 | Token Distribution Side-Channel | v14 Phase 5 |
| L20 | System Prompt Harvester | v14 Phase 3 (index 19) |

The `predetect_result.layers` array in the API response now contains up to 20 entries.
Frontend code iterating `layers` should handle this extended range.

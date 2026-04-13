# Benchmarks Directory

This directory is reserved for storing benchmark model profiles.

Benchmark data is managed through the application:
- Mark a completed test run as a baseline via the UI or API
- Baseline profiles are stored in the `golden_baselines` database table
- The SimilarityEngine uses these baselines for cosine similarity comparison

## File Structure (when populated)

```
benchmarks/
├── gpt-4o.json          # GPT-4o baseline feature vector
├── claude-3.5.json      # Claude 3.5 baseline
└── ...
```

> Note: Currently, baselines are stored in the SQLite database, not as files.
> This directory exists for potential future file-based benchmark exports.

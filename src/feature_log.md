# Feature Log

Version history of `src/features.py`. The feature-engineer agent adds a row whenever the feature set changes. Versioning lives in this table — `make_features` no longer takes a `version` argument.

| Version | Date       | Features Added/Changed       | Feature Count | CV Sharpe | Notes                                                                |
|---------|------------|------------------------------|---------------|-----------|----------------------------------------------------------------------|
| v0      | 2026-04-18 | baseline `_const` stub       | 1             | ~0.0      | Empty scaffold — `make_features` returns only `_const`.              |

## Deviations from agent spec

- `make_features` returns `(X, feature_names, fitted_stats)` — no `y`. Target stays owned by `src/data.compute_targets()` to keep a single source of truth and avoid test-time asymmetry.

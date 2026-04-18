# Feature Log

Version history of `src/features.py`. The feature-engineer agent adds a row whenever the feature set changes. Versioning lives in this table — `make_features` no longer takes a `version` argument.

| Version | Date       | Features Added/Changed       | Feature Count | CV Sharpe | Notes                                                                |
|---------|------------|------------------------------|---------------|-----------|----------------------------------------------------------------------|
| v0      | 2026-04-18 | baseline `_const` stub       | 1             | ~0.0      | Empty scaffold — `make_features` returns only `_const`.              |
| v1      | 2026-04-18 | + fh_return (intraday momentum / mean-reversion) | 2             | 3.06 ± 0.70 | r ≈ -0.07 vs target (mean-reverting). Gate: \|corr\| > 0.03. Lift vs v0: v0 ≈ 0.0 sharpe → v1 = 3.06 (sharpe_raw mean across 5 CV folds). |
| v2      | 2026-04-18 | + yz_vol (Yang-Zhang volatility over bars 0-49) | 3             | 3.18 ± 0.64 | Gate 1 ✓ (\|r\|=0.079, MI=0.0092). Gate 2 ✓ (lift +0.12, 4/5 folds). Gate 3 ✓ (WD = 2% of std). |

## Deviations from agent spec

- `make_features` returns `(X, feature_names, fitted_stats)` — no `y`. Target stays owned by `src/data.compute_targets()` to keep a single source of truth and avoid test-time asymmetry.

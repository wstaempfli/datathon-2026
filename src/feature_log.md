# Feature Log

Version history of `src/features.py`. The feature-engineer agent adds a row whenever the feature set changes. Versioning lives in this table — `make_features` no longer takes a `version` argument.

| Version | Date       | Features Added/Changed       | Feature Count | CV Sharpe | Notes                                                                |
|---------|------------|------------------------------|---------------|-----------|----------------------------------------------------------------------|
| v0      | 2026-04-18 | baseline `_const` stub       | 1             | ~0.0      | Empty scaffold — `make_features` returns only `_const`.              |
| v1      | 2026-04-18 | + fh_return (intraday momentum / mean-reversion) | 2             | 3.06 ± 0.70 | r ≈ -0.07 vs target (mean-reverting). Gate: \|corr\| > 0.03. Lift vs v0: v0 ≈ 0.0 sharpe → v1 = 3.06 (sharpe_raw mean across 5 CV folds). |
| v2      | 2026-04-18 | + yz_vol (Yang-Zhang volatility over bars 0-49) | 3             | 3.18 ± 0.64 | Gate 1 ✓ (\|r\|=0.079, MI=0.0092). Gate 2 ✓ (lift +0.12, 4/5 folds). Gate 3 ✓ (WD = 2% of std). |
| v10     | 2026-04-18 | + 10 Gate-1 PASS features (max_drawdown_fh, garman_klass_vol, rogers_satchell_vol, parkinson_vol, sent_sma10, sent_ema20, sent_wt_decay20, realized_skewness, rsi_14, embed_pca_recency_1 — exact names per `submissions/phase3a_features.txt`). Model swap: Huber → XGBoost L1+L2 (reg_alpha=1.0, reg_lambda=1.0). | 15 | 2.930 ± 0.570 | Winning submit commit 271b2de (`submit: xgboost / price5+gate1top10 / sharpe=2.930`); integration commit on `kenji/main-v2` with subject "integrate: v10 — xgboost_l1l2 / price5+gate1top10 / sharpe=2.930" (resolvable via `git log --oneline --grep '^integrate: v10'`). Phase 3A feature column order frozen; no cross-session fit required. |
| v11     | 2026-04-18 | Feature set unchanged (price5+gate1top10, 15 cols). Model swap: XGBoost L1+L2 → QuantileRegressor(quantile=0.5, alpha=0.01, solver=highs). | 15 | 2.676 ± 0.572 | Median regression directly minimizes absolute deviation (L1), which aligns better with Sharpe's variance penalty than squared loss. Bit-identical to Phase 3A sweep row `20260418T220559_quantile_price5+gate1top10_s268.csv`. Per-fold: 2.689 / 3.175 / 2.744 / 3.162 / 1.607. Final model saved to `models/quantile_final.pkl`. |

## Deviations from agent spec

- `make_features` returns `(X, feature_names, fitted_stats)` — no `y`. Target stays owned by `src/data.compute_targets()` to keep a single source of truth and avoid test-time asymmetry.

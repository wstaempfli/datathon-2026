# Signals Log

The researcher agent logs candidate signals here. The feature-engineer agent reads this file and only implements signals that clear the bar.

## Rules
- **Implement only** signals with `|corr| > 0.03` against `target_return`, OR with explicit mutual-information signal that the researcher has flagged as non-trivial.
- **Date every row** in ISO format (`YYYY-MM-DD`) so stale findings are visible.
- **Status values**: `candidate` (found, not yet turned into a feature), `implemented` (live in `src/features.py`), `rejected` (looked at and dropped — leave the row for the audit trail, note the reason).
- Prefer a single row per signal; update in place rather than appending duplicates.
- If a signal is later invalidated (e.g., leakage discovered, correlation drifts on new data), flip status to `rejected` and add a note — do not delete the row.

## Signals

| Date       | Signal                  | \|corr\| | MI   | Status    | Notes                                                |
|------------|-------------------------|----------|------|-----------|------------------------------------------------------|
| 2026-04-18 | EXAMPLE_first_half_return | 0.00     | n/a  | candidate | Example row showing format — not a real signal yet. |
| 2026-04-18 | intraday_momentum | 0.0712 | 0.0000 | candidate | fh_return = close[bar=49]/open[bar=0]-1. Pearson r=-0.0712 (p=2.4e-02), Spearman=-0.0599, R²=0.005, n=1000. Mean-reverting (opposite of Gao et al. 2018). sklearn MI=0 but linear signal clears threshold — decile 0 mean target +0.76%, decile 9 mean +0.03%. Size short proportional to fh_return. |

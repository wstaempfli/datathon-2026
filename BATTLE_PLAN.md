# Datathon 2026 — 24-Hour Battle Plan

## Context

**Challenge**: Predict stock positions (buy/sell) at the midpoint of simulated 100-bar trading sessions. Scored by Sharpe ratio = mean(PnL) / std(PnL) * 16. We see bars 0-49 (OHLC + headlines) and must output `target_position` per session.

**Why this plan exists**: With 1,000 training sessions, weak signals (r≈0.07), and a noisy headline dataset, this competition rewards disciplined feature engineering and robust evaluation over model complexity. The margin between winning and losing is razor-thin.

---

## Verified Ground Truth (from data analysis)

These numbers are verified — all agents and the main thread confirmed them independently.

### Signal Landscape

| Feature | r with target | p-value | Verdict |
|---------|--------------|---------|---------|
| **revenue event count** | **-0.082** | **0.010** | **Strongest single feature** |
| avg high-low range | +0.078 | 0.013 | Strong |
| return from bar 30 | -0.081 | 0.010 | Strong |
| first-half return | -0.069 | 0.028 | Significant (mean-reversion) |
| volatility (log-ret std) | +0.072 | 0.023 | Significant |
| up-bar ratio | -0.059 | 0.061 | Marginal |
| positive event count | +0.063 | 0.048 | Marginal |
| expansion event count | +0.053 | 0.096 | Weak |
| close rel to range | -0.055 | 0.083 | Weak |
| **keyword sentiment (seen)** | **+0.018** | **0.571** | **DEAD — no signal** |
| main company sentiment | -0.004 | 0.889 | DEAD |
| dollar amounts in headlines | -0.002 | 0.953 | DEAD |
| headline count | -0.011 | 0.729 | DEAD |
| last-10-bar return | -0.025 | 0.436 | DEAD |

### Model Baselines (verified)

| Model | CV Sharpe (mean ± std) | Direction Acc | Notes |
|-------|----------------------|---------------|-------|
| **Always-long (+1)** | **2.766** (exact) | **57.0%** | Floor — must beat this |
| Ridge OHLC (6 feat) | 2.788 ± 0.043 | 57.3% | Stable but tiny edge |
| Ridge OHLC+Headlines | 2.708 ± 0.102 | 57.1% | **Worse** — headlines add noise |
| LightGBM OHLC | 2.433 ± 0.791 | 56.3% | Overfits badly |
| LightGBM OHLC+HL | 2.758 ± 0.845 | 56.8% | Unstable |
| Mean-reversion sign | 0.307 | 46.3% | Terrible alone |
| Momentum sign | -0.307 | 53.7% | Terrible |

### Critical Insight: Headlines

**Seen-only headline sentiment has ZERO predictive power** (r=0.018, p=0.57). The originally-reported r=0.14-0.17 was contaminated by unseen-bar headlines (bars 50-99), which obviously predict second-half returns but are unavailable at test time.

**The ONE useful headline feature**: revenue event count (r=-0.082). More revenue mentions in seen headlines correlates with lower second-half returns. This is the only headline feature worth keeping.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                            │
│                                                              │
│  data/*.parquet                                              │
│       │                                                      │
│       ▼                                                      │
│  src/features.py::build_features(bars, headlines)            │
│       │                                                      │
│       ├──► OHLC Features (6-10 features per session)         │
│       │    • first_half_return, ret_from_30, volatility       │
│       │    • avg_hl_range, up_ratio, close_rel_range          │
│       │    • max_drawdown, vol_trend, price_slope             │
│       │                                                      │
│       ├──► Headline Features (1-3 features per session)      │
│       │    • revenue_event_count                             │
│       │    • positive_event_count (marginal)                 │
│       │    • (Claude API sentiment — moonshot experiment)     │
│       │                                                      │
│       ▼                                                      │
│  Session-level feature matrix (1K×10 train, 10K×10 test)     │
│       │                                                      │
│       ▼                                                      │
│  src/models.py::train_model(X, y)                            │
│       │                                                      │
│       ├──► Ridge (primary — stable, doesn't overfit)         │
│       ├──► Logistic (alternative — native probabilities)     │
│       └──► LightGBM (only if heavily regularized)            │
│       │                                                      │
│       ▼                                                      │
│  Position sizing: sign(prediction) × scale                   │
│       │                                                      │
│       ▼                                                      │
│  src/evaluate.py::cross_validate() — 5-fold, 20 seeds        │
│       │                                                      │
│       ▼                                                      │
│  src/submit.py::generate_submission() → submissions/*.csv    │
└─────────────────────────────────────────────────────────────┘
```

---

## 24-Hour Timeline

### PHASE 1: Foundation (Hours 0–3) — Goal: End-to-end pipeline + first submission

```
PERSON A (Pipeline)              PERSON B (Evaluation)
─────────────────────            ─────────────────────
src/features.py                  src/evaluate.py
  build_features()                 compute_sharpe()
src/models.py                      cross_validate()
  train_model()                    run_baselines()
  predict()                      Submit always-long baseline
src/submit.py                    
  generate_submission()          
  validate_submission()          

PERSON C (Research)              PERSON D (Infrastructure)
─────────────────────            ─────────────────────
notebooks/01_eda.ipynb           scripts/run_pipeline.py
  Verify all signal numbers        (ties features→model→eval→submit)
  Visualize price patterns       scripts/leaderboard.py
  Check for nonlinear effects      (tracks all model runs)
  Test feature interactions      Set up reproducible seeds
                                 Git workflow + branching
```

**Milestone 1**: First submission uploaded (always-long). Pipeline runs end-to-end. Baselines documented.

---

### PHASE 2: Feature Discovery (Hours 3–8) — Goal: Find every usable signal

```
PERSON A (OHLC Features)        PERSON B (Model Variants)
─────────────────────            ─────────────────────
Test each new feature by         Try model variants:
adding to Ridge and checking     • Ridge: alpha sweep {0.01..100}
if CV Sharpe improves:           • Logistic: C sweep, l1/l2
                                 • Elastic Net
New features to test:            • SVR (radial)
• max_drawdown                   
• vol_trend (vol bar25-49 /      Record all in leaderboard
  vol bar0-24)                   Focus on STABLE improvement
• price_slope (linear reg)       
• range_contraction              
• bar49 close-to-open gap        
• OHLC pattern features          
• Interaction: ret × vol         

PERSON C (NLP Moonshot)          PERSON D (Position Sizing)
─────────────────────            ─────────────────────
3-hour HARD TIMEBOX:             Test sizing strategies:
                                 • sign(pred): ±1 always
Option A: Claude API             • 2*P(up)-1 (logistic prob)
  Batch classify ~10K headlines  • Asymmetric: +1 / -0.5
  with Haiku ($2-5)              • Clipped linear
  Check if semantic sentiment    • Quantile-based ±{0.5,1}
  beats keyword matching         
  Decision gate: r>0.05 or stop  For each: run 20-seed stability
                                 test and record mean±std Sharpe
Option B: TF-IDF + SVD           
  Extract 5-10 components        Also test: what if we go long
  Test as features in Ridge      on ALL sessions but with
  Fast to implement              varying magnitude?
                                 
IF NOTHING by Hour 6: STOP.      
Redeploy to OHLC features.      
```

**Milestone 2**: Feature set finalized (expect 6-10 features). Headline verdict final. Position sizing selected. Second submission.

---

### PHASE 3: Optimization (Hours 8–14) — Goal: Maximize Sharpe within stability constraints

```
ALL HANDS TOGETHER:

1. Feature selection (2h)
   • Forward selection: add features one at a time, keep only if improves CV Sharpe
   • Backward elimination: remove features one at a time, keep removal if doesn't hurt
   • Compare both — pick the smaller set if Sharpe is similar

2. Hyperparameter sweep (2h)
   • Ridge: alpha ∈ {0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100}
   • All evaluated with 20-seed stability test
   • Plot: alpha vs mean Sharpe with error bars

3. Ensemble exploration (2h)
   • Ridge + Logistic average
   • Ridge + LightGBM (if LightGBM was stabilized)
   • Weighted blending: optimize on CV
   • Stacked: level-1 predictions → level-2 Ridge
```

**Decision rule**: Only switch from current best model if new model has higher mean Sharpe AND lower std across 20 seeds. If equal, prefer the simpler model.

**Milestone 3**: Top 3 candidate models ranked. Third submission (best stable model).

---

### PHASE 4: Lock Down (Hours 14–20) — Goal: Final model + defensive submissions

```
PERSON A + B (Robustness)        PERSON C + D (Submissions)
─────────────────────            ─────────────────────
100-seed stability test          Generate for ALL model variants:
  on final model                   submissions/v01_always_long_public.csv
                                   submissions/v02_ridge_best_public.csv
Edge case analysis:                submissions/v03_ensemble_public.csv
• Extreme first-half returns       submissions/v04_always_long_private.csv
  (top/bottom 10%)                 submissions/v05_ridge_best_private.csv
• Sessions where model shorts      submissions/v06_ensemble_private.csv
  (are they correct?)            
• High-vol vs low-vol sessions   Run validation checklist on EVERY file
                                 
Ablation study:                  Prepare approach documentation
  Remove each feature, measure   
  Sharpe impact                  
```

**Milestone 4**: Final submission files validated and ready. Fallback (always-long) also ready.

---

### PHASE 5: Submit + Buffer (Hours 20–24)

- Upload primary submission (best stable model)
- Upload fallback (always-long) if allowed
- If time: try ONE stretch experiment (Claude API headlines or neural net)
- Document approach for presentation
- **DO NOT touch the primary submission after Hour 22**

---

## Feature Engineering Playbook

### Tier 1 — Build immediately (verified signal)

```python
# All computed from bars_seen (bar_ix 0-49) only
features = {
    'first_half_return': close[49] / close[0] - 1,           # r=-0.069
    'ret_from_30':       close[49] / close[30] - 1,           # r=-0.081
    'volatility':        std(log(close/open)),                 # r=+0.072
    'avg_hl_range':      mean((high - low) / open),            # r=+0.078
    'up_ratio':          count(close > open) / 50,             # r=-0.059
    'close_rel_range':   (close[49] - min(low)) / (max(high) - min(low)),  # r=-0.055
}
```

### Tier 2 — Test in Phase 2 (hypothesized signal)

```python
{
    'revenue_event_count': count("revenue" or "quarterly" in headline),  # r=-0.082!
    'positive_event_count': count("breakthrough" or "surge" in headline), # r=+0.063
    'max_drawdown':      max peak-to-trough in close prices,
    'vol_trend':         std(log_ret[25:49]) / std(log_ret[0:24]),
    'price_slope':       linear_regression_slope(close),
    'range_contraction': mean(hl_range[-10:]) / mean(hl_range[:10]),
    'gap_mean':          mean(open[i] - close[i-1]),
    'ret_x_vol':         first_half_return * volatility,  # interaction
}
```

### Tier 3 — Moonshot (Claude API, TF-IDF)

Only if Person C finds signal in the 3-hour timebox.

---

## Model Selection Guide

### Primary: Ridge Regression (sign-only sizing)
- **Why**: Most stable model. Doesn't overfit with 1K samples. CV Sharpe 2.79 ± 0.04 across 20 seeds.
- **Config**: `Ridge(alpha=1.0)`, then `target_position = np.sign(model.predict(X_test))`
- **Risk**: Barely beats always-long. Edge is ~0.1 Sharpe points.

### Secondary: Logistic Regression
- **Why**: Native probability outputs enable softer position sizing.
- **Config**: `LogisticRegression(C=1.0, penalty='l2')`, target = sign(return)
- **Sizing**: `target_position = 2 * model.predict_proba(X)[:, 1] - 1`

### Avoid: LightGBM/XGBoost (unless heavily regularized)
- **Why**: Overfits badly with 1K samples. CV Sharpe 2.43 ± 0.79 (huge variance).
- **Only use if**: max_depth=2, n_estimators=20, min_child_samples=50, AND it still beats Ridge.

### Ensemble: Ridge + Logistic average
- **Config**: `position = 0.5 * sign(ridge.predict(X)) + 0.5 * sign(logistic.predict(X))`
- **Test this in Phase 3**.

---

## Position Sizing Decision Tree

```
Is direction accuracy > 57%?
├── NO → Submit always-long (+1 for everything)
├── YES, barely (57-58%)
│   └── Use sign-only: position = sign(prediction)
├── YES, moderate (58-60%)
│   └── Try asymmetric: +1 if pred>0, -0.5 if pred<0
│       (respects positive drift, reduces loss from wrong shorts)
└── YES, strong (>60%)
    └── Scale by confidence: position = prediction * scale_factor
```

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Nothing beats always-long | 40% | Submit always-long as primary. Sharpe 2.77 is a respectable score. |
| LightGBM overfits | 80% | Already confirmed. Use Ridge as primary. Only LightGBM in ensemble with heavy constraints. |
| Headlines waste time | 60% | Hard 3-hour timebox (one person). If no signal by Hour 6, abandon completely. |
| Overfit training data | 30% | 20-seed stability test. Never trust single-seed CV. Cap features at 10. |
| Test distribution differs | 10% | Verified: minimal shift (KS p>0.1 on all features except close_rel). |
| Submission format error | 5% | Automated validation function. Run on EVERY submission. |

---

## Claude API Strategy (Person C, Hours 3–6)

**Goal**: Determine if semantic NLP analysis extracts signal that keyword matching cannot.

**Setup**:
```bash
pip install anthropic
```

**Approach**:
```python
# Use claude-3-5-haiku for cost efficiency
# Batch 10 headlines per request
# Prompt: classify each headline as -2/-1/0/+1/+2 sentiment
# Process ~10K training headlines ≈ 1K requests ≈ $2 ≈ 20 min parallel
```

**Decision gate**: Compute correlation of Claude-classified sentiment (seen headlines only!) with second-half returns. If r < 0.05 or p > 0.10 → STOP. Headlines are fundamentally dead for this task.

**If signal found**: Process 200K test headlines (~$10, ~1 hour). Add as feature. Retrain all models. This becomes the winning edge because other teams won't find it.

**Expected outcome**: 85% chance no signal. 15% chance Claude finds r=0.03-0.06 that keywords missed. Bounded downside (3h of one person + $2-12).

---

## Files to Create/Modify

| File | Owner | Phase | Purpose |
|------|-------|-------|---------|
| `src/features.py` | Person A | 1 | `build_features(bars, headlines) → DataFrame` |
| `src/models.py` | Person A | 1 | `train_model(X, y)`, `predict(model, X)` |
| `src/evaluate.py` | Person B | 1 | `compute_sharpe()`, `cross_validate()`, `run_baselines()` |
| `src/submit.py` | Person B | 1 | `generate_submission()`, `validate_submission()` |
| `scripts/run_pipeline.py` | Person D | 1 | End-to-end: features → model → eval → submit |
| `scripts/leaderboard.py` | Person D | 1 | Track model runs + results |
| `notebooks/01_eda.ipynb` | Person C | 1 | Visualization + verification |
| `notebooks/02_claude_nlp.ipynb` | Person C | 2 | Claude API headline experiment |
| `CLAUDE.md` | — | 1 | Update with final verified numbers |

---

## Visualization Aids for the Team

### Quick reference: What predicts returns?

```
Signal Strength (|r| with target return):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

revenue_events  ████████░░  -0.082  p=0.010  ← HEADLINE (only useful one)
ret_from_30     ████████░░  -0.081  p=0.010
avg_hl_range    ███████░░░  +0.078  p=0.013
volatility      ███████░░░  +0.072  p=0.023
first_half_ret  ██████░░░░  -0.069  p=0.028
positive_events █████░░░░░  +0.063  p=0.048
up_ratio        █████░░░░░  -0.059  p=0.061
close_rel       █████░░░░░  -0.055  p=0.083
                                    ↑
                              p=0.05 cutoff
keyword_sent    █░░░░░░░░░  +0.018  p=0.571  ← DEAD
hl_count        █░░░░░░░░░  -0.011  p=0.729  ← DEAD
dollar_amounts  ░░░░░░░░░░  -0.002  p=0.953  ← DEAD
```

### Model Performance Map

```
                    STABLE ◄──────────────► UNSTABLE
                    │                            │
    HIGH SHARPE     │  ★ Ridge OHLC (2.79±0.04)  │
         ▲          │                            │
         │          │  · Ridge+HL (2.71±0.10)    │
         │          │                            │
         │          │          · LGB+HL (2.76±0.85)
         │          │                            │
    LOW SHARPE      │          · LGB OHLC (2.43±0.79)
         │          │                            │
         ▼     ─────┼────── always-long: 2.77 ───┼──── (this is the floor)
                    │                            │
```

### The Headline Truth

```
SEEN headlines (bars 0-49)          UNSEEN headlines (bars 50-99)
┌─────────────────────┐             ┌─────────────────────┐
│  r = 0.018          │             │  r = 0.244          │
│  p = 0.571          │             │  p = 0.000          │
│                     │             │                     │
│  ❌ NO SIGNAL        │             │  ✓ STRONG SIGNAL     │
│  (this is all we    │             │  (but we can't use  │
│   have at test time)│             │   this at test time)│
└─────────────────────┘             └─────────────────────┘
          │                                   │
          └───── The r=0.17 initially ────────┘
                 reported was mixing both!
```

---

## Verification Plan

After implementation, run these checks before any submission:

1. **Pipeline smoke test**: `python scripts/run_pipeline.py` produces a valid CSV in <60s
2. **CV stability**: 20-seed test shows Sharpe std < 0.10 for primary model
3. **Beats always-long**: CV Sharpe mean > 2.77 for primary model
4. **Submission validation**: correct columns, correct sessions, no NaN/Inf
5. **Position distribution**: check that positions are not all identical (model degeneracy)
6. **Direction accuracy**: > 57% (at least marginally better than base rate)

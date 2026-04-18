"""Render the rule_bmbv2 grid-search train Sharpe as a heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "plots" / "sweep_bmbv2_heatmap.png"

K_FH = [15, 20, 25, 30, 35, 40, 50]
W_BMB = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

grid = np.array([
    [2.848, 3.027, 3.042, 3.018, 2.984, 2.938, 2.844],
    [2.809, 3.021, 3.049, 3.041, 3.008, 2.975, 2.884],
    [2.763, 3.007, 3.053, 3.057, 3.027, 3.005, 2.917],
    [2.722, 2.970, 3.042, 3.065, 3.048, 3.021, 2.942],
    [2.675, 2.909, 2.994, 3.044, 3.052, 3.028, 2.954],
    [2.636, 2.856, 2.934, 3.002, 3.034, 3.026, 2.970],
    [2.568, 2.783, 2.849, 2.902, 2.954, 2.988, 2.981],
])

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="upper")

ax.set_xticks(range(len(W_BMB)))
ax.set_xticklabels([f"{w:.2f}" for w in W_BMB])
ax.set_yticks(range(len(K_FH)))
ax.set_yticklabels(K_FH)
ax.set_xlabel("W_BMB (bmb coefficient)")
ax.set_ylabel("K_FH (fh_return coefficient)")
ax.set_title(
    "rule_bmbv2 grid search — train Sharpe\n"
    "clip(1 - K_FH·fh_return + W_BMB·bmb, 0.2, 2.0)"
)

best_i, best_j = np.unravel_index(np.argmax(grid), grid.shape)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        val = grid[i, j]
        color = "white" if val < grid.mean() else "black"
        weight = "bold" if (i, j) == (best_i, best_j) else "normal"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                color=color, fontsize=9, fontweight=weight)

ax.scatter([best_j], [best_i], s=300, facecolors="none",
           edgecolors="red", linewidths=2.0, label="train-best")
ax.legend(loc="lower right")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("train Sharpe")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")

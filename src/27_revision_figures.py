#!/usr/bin/env python3
"""
27_revision_figures.py — Figure 6 for the IEEE Access revision (Access-2026-33087).

(a) Reliability diagram of MC-dropout uncertainty (R2#4): per-bin predicted sigma
    vs empirical RMSE from result/revision/temporal_revision.json.
(b) Retrieval relevance-threshold sensitivity (R1#3, R2#5): R@1 vs threshold from
    result/revision/retrieval_revision.json.

EXACT data from the frozen result JSONs — nothing hand-entered.
Output: result/revision/fig6.png (and copied next to the manuscript by the caller).
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = Path("/home/wjeong/cc/result/revision")
temporal = json.load(open(RES / "temporal_revision.json"))
retrieval = json.load(open(RES / "retrieval_revision.json"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) reliability diagram
sig = temporal['calibration']['reliability_sigma']
rmse = temporal['calibration']['reliability_rmse']
ax = axes[0]
lim = [min(sig + rmse) * 0.8, max(sig + rmse) * 1.3]
ax.plot(lim, lim, 'k--', lw=1.2, label='Perfect calibration')
ax.plot(sig, rmse, 'o-', color='#1f77b4', lw=2, ms=6, label='MC dropout ($N$=20)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('Predicted uncertainty $\\sigma$ (bin mean)', fontsize=11)
ax.set_ylabel('Empirical RMSE (bin)', fontsize=11)
ence = temporal['calibration']['ence']
rho = temporal['calibration']['spearman_pixel']
ax.set_title(f'(a) Reliability diagram (ENCE = {ence:.2f}, '
             f'Spearman $\\rho$ = {rho:.2f})', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

# (b) threshold sensitivity
ax = axes[1]
styles = {
    'Predicted PM SSIM': ('#d62728', 's-'),
    'Predicted PM Cosine': ('#ff7f0e', 'D-'),
    'Encoder Features (Ours)': ('#2ca02c', 'o-'),
    'Raw IR Pixels (L2)': ('#7f7f7f', '^-'),
    'Pretrained ResNet50': ('#9467bd', 'v-'),
}
ths = sorted(retrieval['by_threshold'].keys())
for name, (color, style) in styles.items():
    vals = [retrieval['by_threshold'][t]['methods'][name]['R@1'] for t in ths]
    ax.plot([float(t) for t in ths], vals, style, color=color, lw=2, ms=6,
            label=name)
ax.set_xlabel('Relevance threshold (SSIM between GT pressure maps)\n'
              '(second line: median relevant items per query)', fontsize=10)
ax.set_ylabel('R@1', fontsize=11)
ax.set_title('(b) Retrieval sensitivity to relevance threshold', fontsize=11,
             fontweight='bold')
med = [retrieval['by_threshold'][t]['median_relevant_per_query'] for t in ths]
ax.set_xticks([float(t) for t in ths])
ax.set_xticklabels([f"{float(t):.2f}\n{round(m)}" for t, m in zip(ths, med)],
                   fontsize=9)
ax.axvline(0.70, color='k', ls=':', lw=1, alpha=0.6)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = RES / "fig6.png"
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")

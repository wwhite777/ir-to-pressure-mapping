#!/usr/bin/env python3
"""
29_revision_fig2.py — regenerate Figure 2 for the revision (Access-2026-33087).

Panels (a)-(c): one example test sequence re-run with the CAUSAL UA-EMA
calibration (validation-split percentile bounds from temporal_revision.json).
Panel (d): accuracy-stability trade-off scatter from table2_revised — the
EXACT numbers of the revised Table 2 (including tuned Kalman filters).

Output: result/revision/fig2.png
"""

import sys
import json
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
m = importlib.import_module('24_revision_temporal_calibration')

import numpy as np
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = m.RESULT_DIR
tj = json.load(open(RES / "temporal_revision.json"))
U_LO, U_HI = tj['ua_ema_calibration_bounds_val_5_95pct']

# ---- model (MC wrapper), identical to 24 ----
base = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
ckpt = torch.load(m.MODEL_DIR / "U-Net(ResNet50)_E50.pth", map_location=m.device)
sd = {k: v for k, v in ckpt['model_state_dict'].items()
      if not k.endswith(('total_ops', 'total_params'))}
base.load_state_dict(sd, strict=False)
mc = m.MCDropoutWrapper(
    smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1), p=0.1)
mc.model.load_state_dict(sd, strict=False)
mc = mc.to(m.device)
torch.manual_seed(m.SEED)

# ---- first test sequence (same split logic as 24) ----
data = m.load_data()
subs = list(set(k[0] for _, _, k in data))
np.random.seed(m.SEED)
np.random.shuffle(subs)
test_subjects = set(subs[:len(subs) // 5])
seqs = {}
for ir, pm, k in data:
    if k[0] in test_subjects:
        seqs.setdefault((k[0], k[1]), []).append((k[2], ir, pm))
for sk in seqs:
    seqs[sk].sort(key=lambda x: x[0])
keys = [k for k in seqs if len(seqs[k]) >= 20][:m.N_SEQ]
ex_key = keys[0]
frames = seqs[ex_key][:30]
tr = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

import cv2
preds, uncs, gts = [], [], []
for _, irp, pmp in frames:
    irt, gt = m.load_sample(irp, pmp, tr)
    passes = m.mc_forward(mc, irt.unsqueeze(0).to(m.device), m.MC_MAX)
    preds.append(cv2.resize(passes.mean(axis=0), (gt.shape[1], gt.shape[0])))
    uncs.append(float(passes.std(axis=0).mean()))
    gts.append(gt)

alphas = m.ua_ema_alphas(uncs, U_LO, U_HI)
fixed = m.filt_ema(preds, m.FIXED_ALPHA)
adaptive = m.filt_ema(preds, alphas)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fr = list(range(len(preds)))

axes[0, 0].plot(fr, uncs, 'b-', lw=2)
axes[0, 0].fill_between(fr, 0, uncs, alpha=0.3)
axes[0, 0].axhline(U_LO, color='gray', ls=':', lw=1)
axes[0, 0].axhline(U_HI, color='gray', ls=':', lw=1,
                   label='Calibration bounds (val 5th/95th pct)')
axes[0, 0].set_xlabel('Frame')
axes[0, 0].set_ylabel('Uncertainty $u_t$')
axes[0, 0].set_title('(a) Prediction Uncertainty', fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(fr, alphas, 'g-', lw=2, label='Adaptive $\\alpha_t$ (causal)')
axes[0, 1].axhline(0.3, color='r', ls='--', lw=2, label='Fixed $\\alpha$=0.3')
axes[0, 1].set_xlabel('Frame')
axes[0, 1].set_ylabel('Alpha')
axes[0, 1].set_title('(b) Smoothing Factor', fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].set_ylim(0, 0.6)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(fr, [p.mean() for p in preds], 'gray', alpha=0.6, lw=1.2, label='Raw')
axes[1, 0].plot(fr, [p.mean() for p in fixed], 'r-', lw=2, label='Fixed EMA')
axes[1, 0].plot(fr, [p.mean() for p in adaptive], 'g-', lw=2, label='Adaptive EMA')
axes[1, 0].plot(fr, [g.mean() for g in gts], 'b--', lw=1.5, label='GT')
axes[1, 0].set_xlabel('Frame')
axes[1, 0].set_ylabel('Mean Pressure')
axes[1, 0].set_title('(c) Temporal Comparison', fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

t2 = tj['table2_revised']
label_style = {
    'no_filter': ('No Filtering', 'gray', 'o'),
    'moving_avg': ('Moving Avg ($k$=5)', 'tab:blue', 's'),
    'fixed_ema': ('Fixed EMA ($\\alpha$=0.3)', 'tab:red', 'D'),
    'kalman_rw': ('Kalman (RW, tuned)', 'tab:purple', '^'),
    'kalman_cv': ('Kalman (CV, tuned)', 'tab:brown', 'v'),
    'ua_ema': ('Adaptive EMA (Ours)', 'tab:green', '*'),
}
for key, (lab, color, marker) in label_style.items():
    axes[1, 1].scatter(t2[key]['ssim'], t2[key]['jitter'], s=170 if key == 'ua_ema' else 90,
                       c=color, marker=marker, edgecolors='black', zorder=3, label=lab)
axes[1, 1].set_xlabel('SSIM (higher better)')
axes[1, 1].set_ylabel('Jitter (lower better)')
axes[1, 1].set_title('(d) Accuracy-Stability Trade-Off (Test Set)', fontweight='bold')
axes[1, 1].legend(fontsize=8, loc='upper left')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Uncertainty-Guided Adaptive EMA (Causal Calibration)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RES / "fig2.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved {RES / 'fig2.png'}  (example sequence: subject {ex_key[0]}, {ex_key[1]})")

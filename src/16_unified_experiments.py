#!/usr/bin/env python3
"""
16_unified_experiments.py - Unified Experiments for Consistent Numbers

Runs ALL experiments on the SAME test data to ensure consistency.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import re

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path("/home/wjeong/cc")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "unified"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCDropoutWrapper(nn.Module):
    """Wrapper that adds MC Dropout to smp model."""
    def __init__(self, model, p=0.1):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        features = self.model.encoder(x)
        features_with_dropout = []
        for i, f in enumerate(features):
            if i >= len(features) - 2:
                f = self.dropout(f)
            features_with_dropout.append(f)
        decoder_output = self.model.decoder(features_with_dropout)
        decoder_output = self.dropout(decoder_output)
        masks = self.model.segmentation_head(decoder_output)
        return masks


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def load_data():
    """Load all IR/PM pairs."""
    pattern = re.compile(
        r"^(?P<subject>\d+)_(?:ir|lr|pm)_(?P<condition>[A-Za-z0-9]+)_image_(?P<frame>\d+)\.png$",
        re.IGNORECASE
    )

    ir_map = {}
    pm_map = {}

    for f in os.listdir(IR_PATH):
        m = pattern.match(f)
        if m:
            key = (int(m.group('subject')), m.group('condition').lower(), int(m.group('frame')))
            ir_map[key] = IR_PATH / f

    for f in os.listdir(PM_PATH):
        m = pattern.match(f)
        if m:
            key = (int(m.group('subject')), m.group('condition').lower(), int(m.group('frame')))
            pm_map[key] = PM_PATH / f

    common_keys = sorted(set(ir_map.keys()) & set(pm_map.keys()))
    return [(ir_map[k], pm_map[k], k) for k in common_keys]


def load_sample(ir_path, pm_path, transform):
    """Load and preprocess a sample."""
    ir_img = Image.open(ir_path).convert('RGB')
    ir_raw = np.array(ir_img)
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_norm = np.power(ir_norm, 0.75)
    ir_norm = np.clip(ir_norm * 255, 0, 255).astype(np.uint8)

    pm_img = Image.open(pm_path)
    pm_raw = np.array(pm_img)
    if len(pm_raw.shape) == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    pm_norm = pm_raw.astype(np.float32) / 255.0

    ir_tensor = transform(ir_norm)
    return ir_tensor, pm_norm


def compute_metrics(pred, gt):
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - gt))
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    ssim_val = ssim(pred, gt, data_range=1.0)
    return {'ssim': ssim_val, 'rmse': rmse, 'mae': mae, 'psnr': psnr}


def compute_jitter(predictions):
    if len(predictions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(predictions)):
        total += np.abs(predictions[i] - predictions[i-1]).mean()
    return total / (len(predictions) - 1)


def main():
    print(f"=" * 70)
    print("UNIFIED EXPERIMENTS FOR CONSISTENT MANUSCRIPT NUMBERS")
    print(f"=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Load base model (without MC Dropout for standard evaluation)
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    base_model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    base_model.load_state_dict(state_dict, strict=False)
    base_model = base_model.to(device)
    base_model.eval()
    print("Base model loaded")

    # MC Dropout model
    mc_model = MCDropoutWrapper(
        smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1),
        p=0.1
    )
    mc_model.model.load_state_dict(state_dict, strict=False)
    mc_model = mc_model.to(device)
    print("MC Dropout model loaded")

    # Load data
    data = load_data()
    print(f"Total data: {len(data)} samples")

    # Split by subject
    all_subjects = list(set(k[0] for _, _, k in data))
    np.random.seed(42)
    np.random.shuffle(all_subjects)

    n_test = len(all_subjects) // 5
    test_subjects = set(all_subjects[:n_test])

    test_data = [(ir, pm, k) for ir, pm, k in data if k[0] in test_subjects]
    print(f"Test data: {len(test_data)} samples from {len(test_subjects)} subjects")

    # Group by condition and sequence
    test_by_cond = {'uncover': [], 'cover1': [], 'cover2': []}
    test_seqs = {}

    for ir, pm, k in test_data:
        cond = k[1]
        if cond in test_by_cond:
            test_by_cond[cond].append((ir, pm, k))
        seq_key = (k[0], k[1])
        if seq_key not in test_seqs:
            test_seqs[seq_key] = []
        test_seqs[seq_key].append((k[2], ir, pm))

    for seq_key in test_seqs:
        test_seqs[seq_key].sort(key=lambda x: x[0])

    print(f"Sequences: {len(test_seqs)}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    # =========================================================================
    # TABLE 1 & 3: Per-condition metrics
    # =========================================================================
    print(f"\n{'='*70}")
    print("TABLE 1 & 3: Per-Condition Evaluation")
    print(f"{'='*70}")

    condition_metrics = {}
    all_metrics = {'ssim': [], 'rmse': [], 'mae': [], 'psnr': []}

    for cond in ['uncover', 'cover1', 'cover2']:
        samples = test_by_cond[cond][:300]
        metrics = {'ssim': [], 'rmse': [], 'mae': [], 'psnr': []}

        for ir_path, pm_path, _ in tqdm(samples, desc=f"{cond}"):
            ir_tensor, pm_gt = load_sample(ir_path, pm_path, transform)
            ir_tensor = ir_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = base_model(ir_tensor).squeeze().cpu().numpy()

            pred = cv2.resize(pred, (pm_gt.shape[1], pm_gt.shape[0]))
            m = compute_metrics(pred, pm_gt)

            for k, v in m.items():
                metrics[k].append(v)
                all_metrics[k].append(v)

        condition_metrics[cond] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in metrics.items()}

    overall = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in all_metrics.items()}

    print(f"\nTABLE 1: U-Net (ResNet50)")
    print(f"  SSIM: {overall['ssim']['mean']:.4f} ± {overall['ssim']['std']:.4f}")
    print(f"  RMSE: {overall['rmse']['mean']:.4f} ± {overall['rmse']['std']:.4f}")
    print(f"  MAE:  {overall['mae']['mean']:.4f} ± {overall['mae']['std']:.4f}")

    print(f"\nTABLE 3: Per-Condition")
    for cond in ['uncover', 'cover1', 'cover2']:
        m = condition_metrics[cond]
        print(f"  {cond:12} SSIM={m['ssim']['mean']:.4f}±{m['ssim']['std']:.3f}  RMSE={m['rmse']['mean']:.4f}±{m['rmse']['std']:.3f}")

    # Compute degradation
    base_ssim = condition_metrics['uncover']['ssim']['mean']
    for cond in ['cover1', 'cover2']:
        drop = (base_ssim - condition_metrics[cond]['ssim']['mean']) / base_ssim * 100
        print(f"  {cond} SSIM drop: {drop:.1f}%")

    # =========================================================================
    # TABLE 2: Temporal filtering
    # =========================================================================
    print(f"\n{'='*70}")
    print("TABLE 2: Temporal Filtering")
    print(f"{'='*70}")

    seq_keys = [k for k in test_seqs.keys() if len(test_seqs[k]) >= 20][:30]

    temporal = {
        'no_filter': {'ssim': [], 'jitter': []},
        'moving_avg': {'ssim': [], 'jitter': []},
        'fixed_ema': {'ssim': [], 'jitter': []},
        'adaptive_ema': {'ssim': [], 'jitter': []},
    }

    fig2_uncertainties = []
    fig2_alphas = []

    for seq_key in tqdm(seq_keys, desc="Sequences"):
        frames = test_seqs[seq_key][:40]

        raw_preds = []
        uncertainties = []
        gts = []

        for _, ir_path, pm_path in frames:
            ir_tensor, pm_gt = load_sample(ir_path, pm_path, transform)
            ir_tensor = ir_tensor.unsqueeze(0).to(device)

            # MC Dropout for uncertainty
            mc_model.eval()
            enable_dropout(mc_model)
            preds_mc = []
            with torch.no_grad():
                for _ in range(20):
                    p = mc_model(ir_tensor).squeeze().cpu().numpy()
                    preds_mc.append(p)

            mean_pred = np.mean(preds_mc, axis=0)
            unc = np.std(preds_mc, axis=0).mean()

            mean_pred = cv2.resize(mean_pred, (pm_gt.shape[1], pm_gt.shape[0]))

            raw_preds.append(mean_pred)
            uncertainties.append(unc)
            gts.append(pm_gt)

        # Compute adaptive alphas
        u_arr = np.array(uncertainties)
        u_min, u_max = u_arr.min(), u_arr.max()
        alphas = [0.5 - ((u - u_min) / (u_max - u_min + 1e-8)) * 0.4 for u in uncertainties]

        fig2_uncertainties.extend(uncertainties)
        fig2_alphas.extend(alphas)

        # Apply filters
        ma_preds = [np.mean(raw_preds[max(0,i-4):i+1], axis=0) for i in range(len(raw_preds))]

        fixed_preds = []
        state = None
        for p in raw_preds:
            state = p.copy() if state is None else 0.3 * p + 0.7 * state
            fixed_preds.append(state.copy())

        adaptive_preds = []
        state = None
        for p, a in zip(raw_preds, alphas):
            state = p.copy() if state is None else a * p + (1 - a) * state
            adaptive_preds.append(state.copy())

        def seq_ssim(preds):
            return np.mean([ssim(np.clip(p, 0, 1), g, data_range=1.0) for p, g in zip(preds, gts)])

        temporal['no_filter']['ssim'].append(seq_ssim(raw_preds))
        temporal['no_filter']['jitter'].append(compute_jitter(raw_preds))
        temporal['moving_avg']['ssim'].append(seq_ssim(ma_preds))
        temporal['moving_avg']['jitter'].append(compute_jitter(ma_preds))
        temporal['fixed_ema']['ssim'].append(seq_ssim(fixed_preds))
        temporal['fixed_ema']['jitter'].append(compute_jitter(fixed_preds))
        temporal['adaptive_ema']['ssim'].append(seq_ssim(adaptive_preds))
        temporal['adaptive_ema']['jitter'].append(compute_jitter(adaptive_preds))

    print(f"\n{'Method':<20} {'SSIM':>10} {'Jitter':>12} {'Reduction':>10}")
    print("-" * 55)

    raw_jitter = np.mean(temporal['no_filter']['jitter'])
    for method, name in [('no_filter', 'No Filtering'), ('moving_avg', 'Moving Avg (k=5)'),
                          ('fixed_ema', 'Fixed EMA (α=0.3)'), ('adaptive_ema', 'Adaptive EMA')]:
        ssim_m = np.mean(temporal[method]['ssim'])
        jitter = np.mean(temporal[method]['jitter'])
        red = (1 - jitter / raw_jitter) * 100 if raw_jitter > 0 else 0
        print(f"{name:<20} {ssim_m:>10.4f} {jitter:>12.6f} {red:>9.1f}%")

    # =========================================================================
    # FIGURE 2
    # =========================================================================
    print(f"\nGenerating Figure 2...")

    # Use first sequence for visualization
    example_key = seq_keys[0]
    ex_frames = test_seqs[example_key][:30]

    ex_preds, ex_uncs, ex_gts = [], [], []
    for _, ir_path, pm_path in ex_frames:
        ir_tensor, pm_gt = load_sample(ir_path, pm_path, transform)
        ir_tensor = ir_tensor.unsqueeze(0).to(device)

        mc_model.eval()
        enable_dropout(mc_model)
        preds_mc = []
        with torch.no_grad():
            for _ in range(20):
                p = mc_model(ir_tensor).squeeze().cpu().numpy()
                preds_mc.append(p)
        mean_pred = np.mean(preds_mc, axis=0)
        unc = np.std(preds_mc, axis=0).mean()
        mean_pred = cv2.resize(mean_pred, (pm_gt.shape[1], pm_gt.shape[0]))

        ex_preds.append(mean_pred)
        ex_uncs.append(unc)
        ex_gts.append(pm_gt)

    ex_u = np.array(ex_uncs)
    ex_alphas = [0.5 - ((u - ex_u.min()) / (ex_u.max() - ex_u.min() + 1e-8)) * 0.4 for u in ex_uncs]

    ex_fixed, state = [], None
    for p in ex_preds:
        state = p.copy() if state is None else 0.3 * p + 0.7 * state
        ex_fixed.append(state.copy())

    ex_adaptive, state = [], None
    for p, a in zip(ex_preds, ex_alphas):
        state = p.copy() if state is None else a * p + (1 - a) * state
        ex_adaptive.append(state.copy())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    frames = list(range(len(ex_preds)))

    axes[0, 0].plot(frames, ex_uncs, 'b-', lw=2)
    axes[0, 0].fill_between(frames, 0, ex_uncs, alpha=0.3)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Uncertainty')
    axes[0, 0].set_title('(a) Prediction Uncertainty', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(frames, ex_alphas, 'g-', lw=2, label='Adaptive α')
    axes[0, 1].axhline(0.3, color='r', ls='--', lw=2, label='Fixed α=0.3')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Alpha')
    axes[0, 1].set_title('(b) Smoothing Factor', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 0.6)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(frames, [p.max() for p in ex_preds], 'gray', alpha=0.5, lw=1, label='Raw')
    axes[1, 0].plot(frames, [p.max() for p in ex_fixed], 'r-', lw=2, label='Fixed EMA')
    axes[1, 0].plot(frames, [p.max() for p in ex_adaptive], 'g-', lw=2, label='Adaptive EMA')
    axes[1, 0].plot(frames, [g.max() for g in ex_gts], 'b--', lw=1.5, label='GT')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Peak Pressure')
    axes[1, 0].set_title('(c) Temporal Comparison', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    methods = ['No Filter', 'Moving Avg', 'Fixed EMA', 'Adaptive EMA']
    jitters = [np.mean(temporal[m]['jitter']) for m in ['no_filter', 'moving_avg', 'fixed_ema', 'adaptive_ema']]
    bars = axes[1, 1].bar(methods, jitters, color=['gray', 'blue', 'red', 'green'], edgecolor='black')
    axes[1, 1].set_ylabel('Jitter')
    axes[1, 1].set_title('(d) Jitter (Dataset Average)', fontweight='bold')
    for bar, j in zip(bars, jitters):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{j:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Uncertainty-Guided Adaptive EMA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "figure2_consistent.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Figure 2")

    # Save results
    results = {
        'table1': overall,
        'table3': condition_metrics,
        'table2': {m: {'ssim': float(np.mean(d['ssim'])), 'jitter': float(np.mean(d['jitter']))} for m, d in temporal.items()},
        'uncertainty_stats': {
            'min': float(np.min(fig2_uncertainties)),
            'max': float(np.max(fig2_uncertainties)),
            'mean': float(np.mean(fig2_uncertainties)),
        },
        'alpha_stats': {
            'min': float(np.min(fig2_alphas)),
            'max': float(np.max(fig2_alphas)),
            'mean': float(np.mean(fig2_alphas)),
        }
    }

    with open(RESULT_DIR / "unified_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULT_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
17_final_figures.py - Generate all final figures for ICMR 2026 submission

Figures:
- Figure 1: Method overview (conceptual diagram - manual in drawio)
- Figure 2: Adaptive EMA demonstration (already generated)
- Figure 3: Qualitative results (IR → Pressure prediction)
- Figure 4: Cross-modal retrieval (already generated)
- Figure 5: Per-condition performance comparison
"""

import os
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
import cv2
import re

# Paths
PROJECT_ROOT = Path("/home/wjeong/cc")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "final"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load unified results
with open(PROJECT_ROOT / "result" / "unified" / "unified_results.json") as f:
    UNIFIED_RESULTS = json.load(f)


def load_data():
    """Load IR and PM data."""
    pattern = re.compile(
        r"^(?P<subject>\d+)_(?:ir|lr|pm)_(?P<condition>[A-Za-z0-9]+)_image_(?P<frame>\d+)\.png$",
        re.IGNORECASE
    )

    def parse_key(filename):
        m = pattern.match(filename)
        if not m:
            return None
        return (int(m.group('subject')), m.group('condition').lower(), int(m.group('frame')))

    ir_files = {}
    pm_files = {}

    for f in os.listdir(IR_PATH):
        key = parse_key(f)
        if key:
            ir_files[key] = IR_PATH / f

    for f in os.listdir(PM_PATH):
        key = parse_key(f)
        if key:
            pm_files[key] = PM_PATH / f

    common_keys = sorted(set(ir_files.keys()) & set(pm_files.keys()))
    return [(ir_files[k], pm_files[k], k) for k in common_keys]


def load_and_preprocess(ir_path, pm_path):
    """Load and preprocess IR and PM images."""
    ir_img = Image.open(ir_path).convert('RGB')
    ir_raw = np.array(ir_img)
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_norm = np.power(ir_norm, 0.75)
    ir_display = np.clip(ir_norm * 255, 0, 255).astype(np.uint8)

    pm_img = Image.open(pm_path)
    pm_raw = np.array(pm_img)
    if len(pm_raw.shape) == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    pm_norm = pm_raw.astype(np.float32) / 255.0

    return ir_display, pm_norm


def create_figure3_qualitative(model, data):
    """Create Figure 3: Qualitative results showing IR → Pressure prediction."""
    print("Creating Figure 3: Qualitative Results...")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    # Select samples from different conditions
    conditions = ['uncover', 'cover1', 'cover2']
    samples_per_condition = 2

    selected = []
    for cond in conditions:
        cond_data = [(ir, pm, k) for ir, pm, k in data if k[1] == cond]
        np.random.seed(42)
        indices = np.random.choice(len(cond_data), min(samples_per_condition, len(cond_data)), replace=False)
        for idx in indices:
            selected.append(cond_data[idx])

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(len(selected), 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1, hspace=0.3)

    for i, (ir_path, pm_path, key) in enumerate(selected):
        ir_display, pm_gt = load_and_preprocess(ir_path, pm_path)

        # Predict
        ir_tensor = transform(ir_display).unsqueeze(0).to(device)
        with torch.no_grad():
            pm_pred = model(ir_tensor).squeeze().cpu().numpy()

        pm_pred = np.clip(pm_pred, 0, 1)

        # Resize prediction to match GT
        pm_pred = cv2.resize(pm_pred, (pm_gt.shape[1], pm_gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Compute metrics
        ssim_val = ssim(pm_gt, pm_pred, data_range=1.0)
        rmse_val = np.sqrt(np.mean((pm_gt - pm_pred) ** 2))

        # Plot IR
        ax_ir = fig.add_subplot(gs[i, 0])
        ax_ir.imshow(ir_display)
        ax_ir.axis('off')
        if i == 0:
            ax_ir.set_title('Input IR', fontsize=12, fontweight='bold')

        condition_labels = {'uncover': 'No Blanket', 'cover1': 'Thin Blanket', 'cover2': 'Thick Blanket'}
        ax_ir.text(-0.15, 0.5, condition_labels.get(key[1], key[1]), transform=ax_ir.transAxes,
                   fontsize=10, fontweight='bold', va='center', rotation=90)

        # Plot GT pressure
        ax_gt = fig.add_subplot(gs[i, 1])
        im_gt = ax_gt.imshow(pm_gt, cmap='jet', vmin=0, vmax=1)
        ax_gt.axis('off')
        if i == 0:
            ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')

        # Plot predicted pressure
        ax_pred = fig.add_subplot(gs[i, 2])
        im_pred = ax_pred.imshow(pm_pred, cmap='jet', vmin=0, vmax=1)
        ax_pred.axis('off')
        if i == 0:
            ax_pred.set_title('Predicted', fontsize=12, fontweight='bold')
        ax_pred.text(1.05, 0.5, f'SSIM: {ssim_val:.3f}\nRMSE: {rmse_val:.3f}',
                    transform=ax_pred.transAxes, fontsize=9, va='center')

    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(im_pred, cax=cbar_ax)
    cbar.set_label('Pressure (normalized)', fontsize=10)

    plt.suptitle('Qualitative Results: IR-to-Pressure Prediction Across Occlusion Conditions',
                fontsize=14, fontweight='bold', y=0.98)

    save_path = RESULT_DIR / "figure3_qualitative.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def create_figure5_conditions():
    """Create Figure 5: Per-condition performance comparison."""
    print("Creating Figure 5: Per-Condition Performance...")

    # Data from unified results
    conditions = ['uncover', 'cover1', 'cover2']
    labels = ['No Blanket', 'Thin Blanket', 'Thick Blanket']

    ssim_means = [UNIFIED_RESULTS['table3'][c]['ssim']['mean'] for c in conditions]
    ssim_stds = [UNIFIED_RESULTS['table3'][c]['ssim']['std'] for c in conditions]

    rmse_means = [UNIFIED_RESULTS['table3'][c]['rmse']['mean'] for c in conditions]
    rmse_stds = [UNIFIED_RESULTS['table3'][c]['rmse']['std'] for c in conditions]

    psnr_means = [UNIFIED_RESULTS['table3'][c]['psnr']['mean'] for c in conditions]
    psnr_stds = [UNIFIED_RESULTS['table3'][c]['psnr']['std'] for c in conditions]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    x = np.arange(len(conditions))
    width = 0.6

    # SSIM
    bars1 = axes[0].bar(x, ssim_means, width, yerr=ssim_stds, capsize=5, color=colors, edgecolor='black')
    axes[0].set_ylabel('SSIM', fontsize=12)
    axes[0].set_title('(a) Structural Similarity', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylim(0.65, 0.80)
    axes[0].axhline(y=ssim_means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline (No Blanket)')

    # Add degradation annotation
    for i, (mean, std) in enumerate(zip(ssim_means, ssim_stds)):
        axes[0].text(i, mean + std + 0.005, f'{mean:.3f}', ha='center', fontsize=9)

    # Calculate degradation
    deg_cover1 = (ssim_means[0] - ssim_means[1]) / ssim_means[0] * 100
    deg_cover2 = (ssim_means[0] - ssim_means[2]) / ssim_means[0] * 100
    axes[0].text(1, 0.67, f'{deg_cover1:+.1f}%', ha='center', fontsize=9, color='#3498db')
    axes[0].text(2, 0.67, f'{deg_cover2:+.1f}%', ha='center', fontsize=9, color='#e74c3c')

    # RMSE
    bars2 = axes[1].bar(x, rmse_means, width, yerr=rmse_stds, capsize=5, color=colors, edgecolor='black')
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('(b) Root Mean Square Error', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylim(0.04, 0.07)
    axes[1].axhline(y=rmse_means[0], color='gray', linestyle='--', alpha=0.5)

    for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
        axes[1].text(i, mean + std + 0.001, f'{mean:.4f}', ha='center', fontsize=9)

    # PSNR
    bars3 = axes[2].bar(x, psnr_means, width, yerr=psnr_stds, capsize=5, color=colors, edgecolor='black')
    axes[2].set_ylabel('PSNR (dB)', fontsize=12)
    axes[2].set_title('(c) Peak Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=10)
    axes[2].set_ylim(24, 27)
    axes[2].axhline(y=psnr_means[0], color='gray', linestyle='--', alpha=0.5)

    for i, (mean, std) in enumerate(zip(psnr_means, psnr_stds)):
        axes[2].text(i, mean + std + 0.05, f'{mean:.2f}', ha='center', fontsize=9)

    plt.suptitle('Performance Across Occlusion Conditions (Dashed line = No Blanket baseline)',
                fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    save_path = RESULT_DIR / "figure5_conditions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def create_figure_temporal_comparison():
    """Create bar chart comparing temporal filtering methods."""
    print("Creating Temporal Filtering Comparison Figure...")

    methods = ['No Filter', 'Moving Avg', 'Fixed EMA', 'Adaptive EMA']
    ssim_vals = [
        UNIFIED_RESULTS['table2']['no_filter']['ssim'],
        UNIFIED_RESULTS['table2']['moving_avg']['ssim'],
        UNIFIED_RESULTS['table2']['fixed_ema']['ssim'],
        UNIFIED_RESULTS['table2']['adaptive_ema']['ssim']
    ]
    jitter_vals = [
        UNIFIED_RESULTS['table2']['no_filter']['jitter'],
        UNIFIED_RESULTS['table2']['moving_avg']['jitter'],
        UNIFIED_RESULTS['table2']['fixed_ema']['jitter'],
        UNIFIED_RESULTS['table2']['adaptive_ema']['jitter']
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = ['#7f8c8d', '#3498db', '#e74c3c', '#2ecc71']
    x = np.arange(len(methods))
    width = 0.6

    # SSIM
    bars1 = ax1.bar(x, ssim_vals, width, color=colors, edgecolor='black')
    ax1.set_ylabel('SSIM', fontsize=12)
    ax1.set_title('(a) Prediction Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylim(0.5, 0.8)

    for i, v in enumerate(ssim_vals):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # Jitter
    bars2 = ax2.bar(x, jitter_vals, width, color=colors, edgecolor='black')
    ax2.set_ylabel('Jitter', fontsize=12)
    ax2.set_title('(b) Temporal Jitter', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.set_ylim(0, 0.03)

    for i, v in enumerate(jitter_vals):
        ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

    # Add reduction percentages
    baseline_jitter = jitter_vals[0]
    for i in range(1, len(jitter_vals)):
        reduction = (baseline_jitter - jitter_vals[i]) / baseline_jitter * 100
        ax2.text(i, jitter_vals[i] / 2, f'-{reduction:.0f}%', ha='center', fontsize=9, fontweight='bold', color='white')

    plt.suptitle('Temporal Filtering Methods Comparison', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = RESULT_DIR / "figure_temporal_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print(f"Generating Final Figures for ICMR 2026")
    print(f"Device: {device}")
    print(f"Output: {RESULT_DIR}")

    # Load model
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded")

    # Load data
    data = load_data()
    print(f"Loaded {len(data)} samples")

    # Generate figures
    create_figure3_qualitative(model, data)
    create_figure5_conditions()
    create_figure_temporal_comparison()

    # Copy existing figures
    import shutil

    src_fig2 = PROJECT_ROOT / "result" / "unified" / "figure2_consistent.png"
    src_fig4 = PROJECT_ROOT / "result" / "figure" / "exp3_retrieval" / "retrieval_with_pressure_maps.png"

    if src_fig2.exists():
        shutil.copy(src_fig2, RESULT_DIR / "figure2_adaptive_ema.png")
        print(f"Copied Figure 2")

    if src_fig4.exists():
        shutil.copy(src_fig4, RESULT_DIR / "figure4_retrieval.png")
        print(f"Copied Figure 4")

    print("\nAll figures saved to:", RESULT_DIR)
    print("\nFigure Summary:")
    print("  - Figure 1: Method overview (create in DrawIO)")
    print("  - Figure 2: Adaptive EMA demonstration")
    print("  - Figure 3: Qualitative results")
    print("  - Figure 4: Cross-modal retrieval")
    print("  - Figure 5: Per-condition performance")
    print("  - Supplementary: Temporal comparison bar chart")


if __name__ == "__main__":
    main()

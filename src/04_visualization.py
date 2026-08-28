#!/usr/bin/env python3
"""
04_visualization.py - Generate Qualitative Figures for ICMR 2026

This script creates:
1. Qualitative figure: IR -> GT -> Pred -> Error (3 rows: Uncover/Cover1/Cover2)
2. Temporal trace figure: Peak pressure at ROI over frames
3. Component contribution visualization
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import re

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cv2

# Paths
PROJECT_ROOT = Path(os.environ.get("CC_PROJECT_ROOT",
                                   Path(__file__).resolve().parent.parent))
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp4_visualization"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IR_PM_Dataset:
    """Simple dataset for visualization."""

    def __init__(self, ir_folder, pm_folder):
        self.ir_folder = Path(ir_folder)
        self.pm_folder = Path(pm_folder)

        pattern = re.compile(
            r"^(?P<subject>\d+)_(?:ir|lr|pm)_(?P<condition>[A-Za-z0-9]+)_image_(?P<frame>\d+)\.(?:png|jpg|jpeg)$",
            re.IGNORECASE
        )

        def parse_key(filename):
            m = pattern.match(filename)
            if not m:
                return None
            return (int(m.group('subject')), m.group('condition').lower(), int(m.group('frame')))

        def build_map(folder):
            mapping = {}
            for f in os.listdir(folder):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                key = parse_key(f)
                if key:
                    mapping[key] = folder / f
            return mapping

        self.ir_map = build_map(self.ir_folder)
        self.pm_map = build_map(self.pm_folder)

        self.common_keys = sorted(set(self.ir_map.keys()) & set(self.pm_map.keys()))

    def get_by_condition(self, condition, n=1):
        """Get samples by condition."""
        samples = [(k, self.ir_map[k], self.pm_map[k])
                   for k in self.common_keys if k[1] == condition]
        return samples[:n]

    def get_sequence(self, subject, condition, max_frames=100):
        """Get a sequence of frames for a subject/condition."""
        samples = [(k, self.ir_map[k], self.pm_map[k])
                   for k in self.common_keys
                   if k[0] == subject and k[1] == condition]
        samples.sort(key=lambda x: x[0][2])  # Sort by frame number
        return samples[:max_frames]


def load_and_preprocess(ir_path, pm_path, model, device):
    """Load images and generate prediction."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    # Load IR
    ir_raw = np.array(Image.open(ir_path).convert('RGB'))
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_norm = np.power(ir_norm, 0.75)
    ir_norm = np.clip(ir_norm * 255, 0, 255).astype(np.uint8)
    ir_tensor = transform(ir_norm).unsqueeze(0).to(device)

    # Load GT
    pm_raw = np.array(Image.open(pm_path))
    if len(pm_raw.shape) == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    pm_norm = pm_raw.astype(np.float32) / 255.0

    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(ir_tensor)
        pred = torch.clamp(pred, 0, 1)
        pred = pred.squeeze().cpu().numpy()

    # Resize prediction to match GT
    pred = cv2.resize(pred, (pm_norm.shape[1], pm_norm.shape[0]))

    return ir_raw, pm_norm, pred


def create_qualitative_figure(dataset, model, device, save_path):
    """
    Create the main qualitative figure:
    3 rows (Uncover, Cover1, Cover2) x 5 columns (IR, GT, Pred, Error, Temporal)
    """
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(3, 5, width_ratios=[1, 1, 1, 1, 1.2], hspace=0.3, wspace=0.2)

    conditions = ['uncover', 'cover1', 'cover2']
    condition_labels = ['Uncover', 'Cover 1', 'Cover 2']

    # Custom colormap for pressure
    pressure_cmap = plt.cm.jet

    # Custom colormap for error (blue-white-red)
    error_cmap = LinearSegmentedColormap.from_list('error', ['blue', 'white', 'red'])

    for row, (condition, label) in enumerate(zip(conditions, condition_labels)):
        samples = dataset.get_by_condition(condition, n=1)
        if not samples:
            continue

        key, ir_path, pm_path = samples[0]
        ir_raw, pm_norm, pred = load_and_preprocess(ir_path, pm_path, model, device)

        # Compute error
        error = pred - pm_norm

        # Column 0: IR input
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(ir_raw)
        ax.set_title('IR Input' if row == 0 else '', fontsize=10)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Column 1: GT pressure
        ax = fig.add_subplot(gs[row, 1])
        im = ax.imshow(pm_norm, cmap=pressure_cmap, vmin=0, vmax=1)
        ax.set_title('Ground Truth' if row == 0 else '', fontsize=10)
        ax.axis('off')

        # Column 2: Prediction
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(pred, cmap=pressure_cmap, vmin=0, vmax=1)
        ax.set_title('Prediction' if row == 0 else '', fontsize=10)
        ax.axis('off')

        # Column 3: Error map
        ax = fig.add_subplot(gs[row, 3])
        im_err = ax.imshow(error, cmap=error_cmap, vmin=-0.3, vmax=0.3)
        ax.set_title('Error Map' if row == 0 else '', fontsize=10)
        ax.axis('off')

        # Column 4: Temporal trace (mini-plot)
        ax = fig.add_subplot(gs[row, 4])

        # Get sequence for temporal trace
        sequence = dataset.get_sequence(key[0], condition, max_frames=50)
        if len(sequence) > 10:
            frames = []
            gt_peaks = []
            pred_peaks = []

            for seq_key, seq_ir, seq_pm in sequence:
                _, pm, pr = load_and_preprocess(seq_ir, seq_pm, model, device)
                frames.append(seq_key[2])
                # Peak pressure in center region (approximating sacrum/body center)
                h, w = pm.shape
                roi = pm[h//3:2*h//3, w//3:2*w//3]
                roi_pred = pr[h//3:2*h//3, w//3:2*w//3]
                gt_peaks.append(roi.max())
                pred_peaks.append(roi_pred.max())

            ax.plot(frames, gt_peaks, 'b-', label='GT', linewidth=1.5)
            ax.plot(frames, pred_peaks, 'r--', label='Pred', linewidth=1.5)
            ax.set_xlabel('Frame', fontsize=8)
            ax.set_ylabel('Peak Pressure', fontsize=8)
            ax.set_title('Temporal Trace' if row == 0 else '', fontsize=10)
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(labelsize=7)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'Insufficient\nframes', ha='center', va='center', fontsize=9)
            ax.axis('off')

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.48, 0.02, 0.15, 0.015])
    cbar1 = plt.colorbar(im, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Pressure', fontsize=8)
    cbar1.ax.tick_params(labelsize=7)

    cbar_ax2 = fig.add_axes([0.66, 0.02, 0.15, 0.015])
    cbar2 = plt.colorbar(im_err, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Error', fontsize=8)
    cbar2.ax.tick_params(labelsize=7)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved qualitative figure to {save_path}")
    plt.close()


def create_temporal_comparison_figure(dataset, model, device, save_path):
    """
    Create temporal trace comparison: Raw U-Net vs EMA-smoothed.
    Shows jitter reduction with EMA.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    conditions = ['uncover', 'cover1', 'cover2']
    condition_labels = ['Uncover', 'Cover 1', 'Cover 2']

    for ax, condition, label in zip(axes, conditions, condition_labels):
        # Find a subject with this condition
        subjects = list(set(k[0] for k in dataset.common_keys if k[1] == condition))
        if not subjects:
            continue

        subject = subjects[0]
        sequence = dataset.get_sequence(subject, condition, max_frames=80)

        if len(sequence) < 20:
            continue

        frames = []
        raw_peaks = []

        for seq_key, seq_ir, seq_pm in sequence:
            _, pm, pr = load_and_preprocess(seq_ir, seq_pm, model, device)
            frames.append(seq_key[2])
            h, w = pr.shape
            roi = pr[h//3:2*h//3, w//3:2*w//3]
            raw_peaks.append(roi.max())

        # Apply EMA
        alpha = 0.3
        ema_peaks = []
        ema_state = None
        for p in raw_peaks:
            if ema_state is None:
                ema_state = p
            else:
                ema_state = alpha * p + (1 - alpha) * ema_state
            ema_peaks.append(ema_state)

        ax.plot(frames, raw_peaks, 'r-', alpha=0.7, label='Raw U-Net', linewidth=1)
        ax.plot(frames, ema_peaks, 'b-', label='EMA Smoothed', linewidth=1.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Peak Pressure')
        ax.set_title(f'{label}')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)

        # Compute jitter reduction
        raw_jitter = np.mean(np.abs(np.diff(raw_peaks)))
        ema_jitter = np.mean(np.abs(np.diff(ema_peaks)))
        reduction = (1 - ema_jitter / raw_jitter) * 100
        ax.text(0.95, 0.05, f'Jitter: -{reduction:.1f}%',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Temporal Stability: Raw Prediction vs EMA Smoothing', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved temporal comparison figure to {save_path}")
    plt.close()


def create_component_ablation_figure(save_path):
    """
    Create bar chart showing component contributions.
    Uses placeholder data - should be updated with actual ablation results.
    """
    # Placeholder data - replace with actual results from ablation study
    components = ['U-Net\nOnly', '+EMA', '+Morph', '+Ridge', 'Full\nPipeline']
    ssim_values = [0.82, 0.84, 0.85, 0.86, 0.88]
    rmse_values = [0.085, 0.078, 0.075, 0.072, 0.068]
    jitter_values = [0.025, 0.012, 0.011, 0.011, 0.010]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(components))
    width = 0.6

    # SSIM
    bars1 = axes[0].bar(x, ssim_values, width, color='steelblue', edgecolor='black')
    axes[0].set_ylabel('SSIM')
    axes[0].set_title('Structural Similarity')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(components, fontsize=9)
    axes[0].set_ylim(0.75, 0.95)
    axes[0].bar_label(bars1, fmt='%.3f', fontsize=8)

    # RMSE
    bars2 = axes[1].bar(x, rmse_values, width, color='coral', edgecolor='black')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Root Mean Square Error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(components, fontsize=9)
    axes[1].set_ylim(0, 0.1)
    axes[1].bar_label(bars2, fmt='%.3f', fontsize=8)

    # Jitter
    bars3 = axes[2].bar(x, jitter_values, width, color='seagreen', edgecolor='black')
    axes[2].set_ylabel('Jitter (MAD)')
    axes[2].set_title('Temporal Jitter')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(components, fontsize=9)
    axes[2].set_ylim(0, 0.03)
    axes[2].bar_label(bars3, fmt='%.3f', fontsize=8)

    plt.suptitle('Ablation Study: Component Contributions', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved component ablation figure to {save_path}")
    plt.close()


def main():
    print(f"ICMR 2026 - Visualization Generation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Load model
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    # Filter out thop profiling keys
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Dataset
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH)
    print(f"Dataset: {len(dataset.common_keys)} samples")

    # Create figures
    print("\nGenerating qualitative figure...")
    create_qualitative_figure(dataset, model, device, RESULT_DIR / "qualitative_figure.png")

    print("\nGenerating temporal comparison figure...")
    create_temporal_comparison_figure(dataset, model, device, RESULT_DIR / "temporal_comparison.png")

    print("\nGenerating component ablation figure...")
    create_component_ablation_figure(RESULT_DIR / "component_ablation.png")

    print(f"\nAll figures saved to: {RESULT_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

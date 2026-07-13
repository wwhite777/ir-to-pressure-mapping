#!/usr/bin/env python3
"""
06_uncertainty_visualization.py - Generate Uncertainty Visualization for ICMR 2026

This script creates:
1. Uncertainty map visualization (showing where model is uncertain)
2. Adaptive alpha visualization (showing how smoothing adapts)
3. Comparison: Fixed EMA vs Adaptive EMA with uncertainty overlay
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import cv2

# Paths
PROJECT_ROOT = Path("/home/wjeong/cc")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp2_uncertainty_ema"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_dropout(model):
    """Enable dropout layers during evaluation for MC Dropout."""
    enabled_count = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()
            enabled_count += 1
    return enabled_count


class MCDropoutWrapper(nn.Module):
    """
    Wrapper that adds MC Dropout to any model.
    Injects Dropout2d after encoder features before feeding to decoder.
    """
    def __init__(self, model, p=0.1):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout2d(p=p)
        self.p = p

    def forward(self, x):
        # Get encoder features
        features = self.model.encoder(x)

        # Apply dropout to bottleneck features (deepest layers)
        features_with_dropout = []
        for i, f in enumerate(features):
            if i >= len(features) - 2:  # Apply dropout to deepest 2 layers
                f = self.dropout(f)
            features_with_dropout.append(f)

        # Decode with dropout-applied features
        decoder_output = self.model.decoder(features_with_dropout)

        # Apply dropout before segmentation head
        decoder_output = self.dropout(decoder_output)

        # Final segmentation
        masks = self.model.segmentation_head(decoder_output)
        return masks

    def encoder(self, x):
        return self.model.encoder(x)


def mc_dropout_predict(model, x, n_samples=20):
    """Perform MC Dropout inference."""
    model.eval()
    enable_dropout(model)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)

    return mean_pred, uncertainty


def load_sample(ir_path, pm_path, transform):
    """Load a single IR/PM sample."""
    ir_img = Image.open(ir_path).convert('RGB')
    ir_raw = np.array(ir_img)

    # Normalize
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_norm = np.power(ir_norm, 0.75)
    ir_norm = np.clip(ir_norm * 255, 0, 255).astype(np.uint8)

    pm_img = Image.open(pm_path)
    pm_raw = np.array(pm_img)
    if len(pm_raw.shape) == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    pm_norm = pm_raw.astype(np.float32) / 255.0

    ir_tensor = transform(ir_norm)

    return ir_raw, ir_tensor, pm_norm


def create_uncertainty_visualization(model, ir_path, pm_path, transform, save_path):
    """Create visualization showing prediction uncertainty."""
    ir_raw, ir_tensor, pm_norm = load_sample(ir_path, pm_path, transform)

    ir_input = ir_tensor.unsqueeze(0).to(device)

    # Get prediction and uncertainty
    mean_pred, uncertainty = mc_dropout_predict(model, ir_input, n_samples=20)

    mean_pred = mean_pred.squeeze().cpu().numpy()
    uncertainty = uncertainty.squeeze().cpu().numpy()

    # Resize to match GT
    mean_pred = cv2.resize(mean_pred, (pm_norm.shape[1], pm_norm.shape[0]))
    uncertainty = cv2.resize(uncertainty, (pm_norm.shape[1], pm_norm.shape[0]))

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Row 1: IR, GT, Prediction
    axes[0, 0].imshow(ir_raw)
    axes[0, 0].set_title('IR Input', fontsize=11)
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(pm_norm, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth', fontsize=11)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(mean_pred, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title('Mean Prediction (MC Dropout)', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Row 2: Uncertainty, Error, Uncertainty vs Error correlation
    im3 = axes[1, 0].imshow(uncertainty, cmap='hot', vmin=0, vmax=uncertainty.max())
    axes[1, 0].set_title('Prediction Uncertainty', fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    error = np.abs(mean_pred - pm_norm)
    im4 = axes[1, 1].imshow(error, cmap='Reds', vmin=0, vmax=0.3)
    axes[1, 1].set_title('Absolute Error', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Scatter plot: uncertainty vs error
    axes[1, 2].scatter(uncertainty.flatten()[::10], error.flatten()[::10],
                       alpha=0.3, s=1, c='blue')
    axes[1, 2].set_xlabel('Uncertainty', fontsize=10)
    axes[1, 2].set_ylabel('Absolute Error', fontsize=10)
    axes[1, 2].set_title('Uncertainty-Error Correlation', fontsize=11)

    # Compute correlation
    corr = np.corrcoef(uncertainty.flatten(), error.flatten())[0, 1]
    axes[1, 2].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Uncertainty Estimation via MC Dropout', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

    return uncertainty, error


def create_adaptive_alpha_visualization(model, dataset_samples, transform, save_path):
    """
    Create visualization showing how adaptive alpha changes with uncertainty.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Process a sequence
    predictions = []
    uncertainties = []
    gts = []

    for ir_path, pm_path in dataset_samples[:30]:
        ir_raw, ir_tensor, pm_norm = load_sample(ir_path, pm_path, transform)
        ir_input = ir_tensor.unsqueeze(0).to(device)

        mean_pred, unc = mc_dropout_predict(model, ir_input, n_samples=10)
        predictions.append(mean_pred.squeeze().cpu().numpy())
        uncertainties.append(unc.squeeze().cpu().numpy().mean())
        gts.append(pm_norm)

    # Compute adaptive alphas
    u_min, u_max = min(uncertainties), max(uncertainties)
    u_range = u_max - u_min + 1e-8

    alpha_min, alpha_max = 0.1, 0.5
    alphas = []
    for u in uncertainties:
        u_norm = (u - u_min) / u_range
        alpha = alpha_max - u_norm * (alpha_max - alpha_min)
        alphas.append(alpha)

    # Apply fixed and adaptive EMA
    fixed_alpha = 0.3
    fixed_ema = []
    adaptive_ema = []

    fixed_state = None
    adaptive_state = None

    for i, pred in enumerate(predictions):
        # Extract peak pressure for plotting
        h, w = pred.shape
        roi = pred[h//3:2*h//3, w//3:2*w//3]
        peak = roi.max()

        if fixed_state is None:
            fixed_state = peak
            adaptive_state = peak
        else:
            fixed_state = fixed_alpha * peak + (1 - fixed_alpha) * fixed_state
            adaptive_state = alphas[i] * peak + (1 - alphas[i]) * adaptive_state

        fixed_ema.append(fixed_state)
        adaptive_ema.append(adaptive_state)

    raw_peaks = [p[p.shape[0]//3:2*p.shape[0]//3, p.shape[1]//3:2*p.shape[1]//3].max()
                 for p in predictions]
    gt_peaks = [g[g.shape[0]//3:2*g.shape[0]//3, g.shape[1]//3:2*g.shape[1]//3].max()
                for g in gts]

    frames = list(range(len(predictions)))

    # Plot 1: Uncertainty over time
    axes[0, 0].plot(frames, uncertainties, 'b-', linewidth=2)
    axes[0, 0].fill_between(frames, 0, uncertainties, alpha=0.3)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Mean Uncertainty')
    axes[0, 0].set_title('(a) Prediction Uncertainty Over Time')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Adaptive alpha over time
    axes[0, 1].plot(frames, alphas, 'g-', linewidth=2, label='Adaptive α')
    axes[0, 1].axhline(y=fixed_alpha, color='r', linestyle='--', linewidth=2, label=f'Fixed α={fixed_alpha}')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Alpha (α)')
    axes[0, 1].set_title('(b) EMA Smoothing Factor')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 0.6)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Raw vs Fixed EMA vs Adaptive EMA
    axes[1, 0].plot(frames, raw_peaks, 'gray', alpha=0.5, linewidth=1, label='Raw')
    axes[1, 0].plot(frames, fixed_ema, 'r-', linewidth=2, label='Fixed EMA')
    axes[1, 0].plot(frames, adaptive_ema, 'g-', linewidth=2, label='Adaptive EMA')
    axes[1, 0].plot(frames, gt_peaks, 'b--', linewidth=1.5, label='GT')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Peak Pressure')
    axes[1, 0].set_title('(c) Temporal Smoothing Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Jitter comparison
    raw_jitter = np.mean(np.abs(np.diff(raw_peaks)))
    fixed_jitter = np.mean(np.abs(np.diff(fixed_ema)))
    adaptive_jitter = np.mean(np.abs(np.diff(adaptive_ema)))

    methods = ['Raw', 'Fixed EMA', 'Adaptive EMA']
    jitters = [raw_jitter, fixed_jitter, adaptive_jitter]
    colors = ['gray', 'red', 'green']

    bars = axes[1, 1].bar(methods, jitters, color=colors, edgecolor='black')
    axes[1, 1].set_ylabel('Mean Absolute Jitter')
    axes[1, 1].set_title('(d) Jitter Comparison')

    for bar, j in zip(bars, jitters):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{j:.4f}', ha='center', va='bottom', fontsize=10)

    # Add reduction percentages
    fixed_reduction = (1 - fixed_jitter / raw_jitter) * 100
    adaptive_reduction = (1 - adaptive_jitter / raw_jitter) * 100
    axes[1, 1].text(1, fixed_jitter/2, f'-{fixed_reduction:.0f}%', ha='center', fontsize=9, color='white', fontweight='bold')
    axes[1, 1].text(2, adaptive_jitter/2, f'-{adaptive_reduction:.0f}%', ha='center', fontsize=9, color='white', fontweight='bold')

    plt.suptitle('Uncertainty-Guided Adaptive EMA Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print(f"ICMR 2026 - Uncertainty Visualization")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Load model
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    base_model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    base_model.load_state_dict(state_dict, strict=False)

    # Wrap with MC Dropout for uncertainty estimation
    model = MCDropoutWrapper(base_model, p=0.1)
    model = model.to(device)
    print("Model loaded successfully with MC Dropout wrapper")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    # Find samples
    pattern = re.compile(
        r"^(?P<subject>\d+)_(?:ir|lr|pm)_(?P<condition>[A-Za-z0-9]+)_image_(?P<frame>\d+)\.png$",
        re.IGNORECASE
    )

    ir_files = sorted([f for f in os.listdir(IR_PATH) if pattern.match(f)])

    # Get samples for visualization
    samples = []
    for f in ir_files[:50]:
        m = pattern.match(f)
        if m:
            subject = m.group('subject')
            condition = m.group('condition')
            frame = m.group('frame')
            ir_path = IR_PATH / f
            pm_name = f.replace('_ir_', '_pm_').replace('_lr_', '_pm_')
            pm_path = PM_PATH / pm_name
            if pm_path.exists():
                samples.append((str(ir_path), str(pm_path)))

    if not samples:
        print("No samples found!")
        sys.exit(1)

    print(f"Found {len(samples)} samples")

    # Create uncertainty visualization for a single sample
    print("\nGenerating uncertainty visualization...")
    create_uncertainty_visualization(
        model, samples[0][0], samples[0][1], transform,
        RESULT_DIR / "uncertainty_map.png"
    )

    # Create adaptive alpha visualization
    print("\nGenerating adaptive alpha visualization...")
    create_adaptive_alpha_visualization(
        model, samples, transform,
        RESULT_DIR / "adaptive_ema_analysis.png"
    )

    print(f"\nAll figures saved to: {RESULT_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

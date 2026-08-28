#!/usr/bin/env python3
"""
08_qualitative_figure.py - Generate qualitative results figure with actual data

Creates a figure showing IR Input, Ground Truth, Prediction, and Error Map
for three occlusion conditions (Uncover, Cover1, Cover2).
"""

import os
import numpy as np
from pathlib import Path
from datetime import datetime
import re

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

# Paths
PROJECT_ROOT = Path(os.environ.get("CC_PROJECT_ROOT",
                                   Path(__file__).resolve().parent.parent))
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp4_qualitative"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load trained U-Net model."""
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_sample(ir_path, pm_path, transform):
    """Load and preprocess a single sample."""
    # Load IR
    ir_img = Image.open(ir_path).convert('RGB')
    ir_raw = np.array(ir_img)

    # Normalize IR for display
    ir_display = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_display = np.power(ir_display, 0.75)
    ir_display = np.clip(ir_display * 255, 0, 255).astype(np.uint8)

    # Load PM
    pm_img = Image.open(pm_path)
    pm_raw = np.array(pm_img)
    if len(pm_raw.shape) == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    pm_norm = pm_raw.astype(np.float32) / 255.0

    # Transform for model
    ir_tensor = transform(ir_display)

    return ir_display, ir_tensor, pm_norm


def find_samples_by_condition(ir_folder, pm_folder):
    """Find samples for each condition."""
    pattern = re.compile(
        r"^(?P<subject>\d+)_(?:ir|lr)_(?P<condition>[A-Za-z0-9]+)_image_(?P<frame>\d+)\.png$",
        re.IGNORECASE
    )

    samples = {'uncover': [], 'cover1': [], 'cover2': []}

    for f in os.listdir(ir_folder):
        m = pattern.match(f)
        if m:
            condition = m.group('condition').lower()
            if condition in samples:
                ir_path = ir_folder / f
                pm_name = f.replace('_ir_', '_pm_').replace('_lr_', '_pm_')
                pm_path = pm_folder / pm_name
                if pm_path.exists():
                    samples[condition].append((str(ir_path), str(pm_path)))

    return samples


def create_qualitative_figure(model, samples, transform, save_path):
    """Create qualitative results figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.15, wspace=0.05)

    conditions = ['uncover', 'cover1', 'cover2']
    condition_labels = ['Uncover', 'Cover1 (thin)', 'Cover2 (thick)']

    for row, (cond, label) in enumerate(zip(conditions, condition_labels)):
        if not samples[cond]:
            continue

        # Pick a good sample (middle of the list)
        idx = len(samples[cond]) // 2
        ir_path, pm_path = samples[cond][idx]

        # Load and predict
        ir_display, ir_tensor, pm_gt = load_sample(ir_path, pm_path, transform)

        with torch.no_grad():
            ir_input = ir_tensor.unsqueeze(0).to(device)
            pred = model(ir_input).squeeze().cpu().numpy()

        # Resize prediction to match GT
        pred = cv2.resize(pred, (pm_gt.shape[1], pm_gt.shape[0]))
        pred = np.clip(pred, 0, 1)

        # Compute error
        error = np.abs(pred - pm_gt)

        # Resize IR for display
        ir_display_resized = cv2.resize(ir_display, (pm_gt.shape[1], pm_gt.shape[0]))

        # Plot IR Input
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(ir_display_resized)
        ax1.set_xticks([])
        ax1.set_yticks([])
        if row == 0:
            ax1.set_title('IR Input', fontsize=12, fontweight='bold')
        ax1.set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=10)

        # Plot Ground Truth
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(pm_gt, cmap='jet', vmin=0, vmax=1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        if row == 0:
            ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')

        # Plot Prediction
        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(pred, cmap='jet', vmin=0, vmax=1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        if row == 0:
            ax3.set_title('Prediction', fontsize=12, fontweight='bold')

        # Plot Error Map
        ax4 = fig.add_subplot(gs[row, 3])
        im4 = ax4.imshow(error, cmap='hot', vmin=0, vmax=0.3)
        ax4.set_xticks([])
        ax4.set_yticks([])
        if row == 0:
            ax4.set_title('Error Map', fontsize=12, fontweight='bold')

        # Add metrics text
        ssim = cv2.matchTemplate(pred.astype(np.float32), pm_gt.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0]
        rmse = np.sqrt(np.mean((pred - pm_gt) ** 2))
        ax4.text(1.05, 0.5, f'RMSE:\n{rmse:.4f}', transform=ax4.transAxes,
                fontsize=9, verticalalignment='center')

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.25, 0.02, 0.25, 0.015])
    cbar1 = plt.colorbar(im2, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Pressure (normalized)', fontsize=9)

    cbar_ax2 = fig.add_axes([0.55, 0.02, 0.25, 0.015])
    cbar2 = plt.colorbar(im4, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Absolute Error', fontsize=9)

    plt.suptitle('Qualitative Results Across Occlusion Conditions', fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print(f"Generating Qualitative Figure")
    print(f"Device: {device}")

    # Load model
    model = load_model()
    print("Model loaded")

    # Setup transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((192, 96))
    ])

    # Find samples
    samples = find_samples_by_condition(IR_PATH, PM_PATH)
    for cond, s in samples.items():
        print(f"  {cond}: {len(s)} samples")

    # Create figure
    create_qualitative_figure(model, samples, transform, RESULT_DIR / "qualitative_results.png")

    print(f"\nSaved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
12_per_condition_eval.py - Per-Occlusion Condition Evaluation

Evaluates model performance separately for each condition:
- Uncover (no occlusion)
- Cover1 (thin blanket)
- Cover2 (thick blanket)

This validates the "Occlusion-Invariant" claim.
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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2

# Paths
PROJECT_ROOT = Path(os.environ.get("CC_PROJECT_ROOT",
                                   Path(__file__).resolve().parent.parent))
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp6_per_condition"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IR_PM_Dataset(Dataset):
    """Dataset for IR to Pressure Map."""

    def __init__(self, ir_folder, pm_folder, transform):
        self.ir_folder = ir_folder
        self.pm_folder = pm_folder
        self.transform = transform

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
                    mapping[key] = os.path.join(folder, f)
            return mapping

        ir_map = build_map(str(self.ir_folder))
        pm_map = build_map(str(self.pm_folder))

        common_keys = sorted(set(ir_map.keys()) & set(pm_map.keys()))
        self.pairs = [(ir_map[k], pm_map[k], k) for k in common_keys]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ir_path, pm_path, key = self.pairs[idx]

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

        ir_tensor = self.transform(ir_norm)
        pm_tensor = torch.from_numpy(pm_norm).unsqueeze(0)

        return ir_tensor, pm_tensor, key


def compute_metrics(pred, gt):
    """Compute all metrics for a single prediction."""
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - gt))
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    ssim_val = ssim(pred, gt, data_range=1.0)

    return {
        'ssim': ssim_val,
        'rmse': rmse,
        'mae': mae,
        'psnr': psnr
    }


def main():
    print(f"ICMR 2026 - Per-Condition Evaluation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

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

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH, transform)
    print(f"Dataset: {len(dataset)} samples")

    # Group samples by condition
    condition_indices = {'uncover': [], 'cover1': [], 'cover2': []}
    for idx, (_, _, key) in enumerate(dataset.pairs):
        cond = key[1]
        if cond in condition_indices:
            condition_indices[cond].append(idx)

    print("\nSamples per condition:")
    for cond, indices in condition_indices.items():
        print(f"  {cond}: {len(indices)}")

    # Evaluate each condition
    results = {}

    for condition, indices in condition_indices.items():
        print(f"\nEvaluating {condition}...")

        # Use subset for efficiency
        eval_indices = indices[:500] if len(indices) > 500 else indices

        metrics_list = {'ssim': [], 'rmse': [], 'mae': [], 'psnr': []}

        with torch.no_grad():
            for idx in tqdm(eval_indices, desc=condition):
                ir, pm, key = dataset[idx]
                ir = ir.unsqueeze(0).to(device)

                pred = model(ir).squeeze().cpu().numpy()
                gt = pm.squeeze().numpy()

                # Resize pred to match gt
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

                m = compute_metrics(pred, gt)
                for k, v in m.items():
                    metrics_list[k].append(v)

        # Compute averages
        results[condition] = {
            'ssim': float(np.mean(metrics_list['ssim'])),
            'ssim_std': float(np.std(metrics_list['ssim'])),
            'rmse': float(np.mean(metrics_list['rmse'])),
            'rmse_std': float(np.std(metrics_list['rmse'])),
            'mae': float(np.mean(metrics_list['mae'])),
            'mae_std': float(np.std(metrics_list['mae'])),
            'psnr': float(np.mean(metrics_list['psnr'])),
            'psnr_std': float(np.std(metrics_list['psnr'])),
            'n_samples': len(eval_indices)
        }

    # Print results table
    print(f"\n{'='*80}")
    print("PER-CONDITION EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Condition':<12} {'SSIM':>12} {'RMSE':>12} {'MAE':>12} {'PSNR':>12}")
    print("-" * 80)

    for cond in ['uncover', 'cover1', 'cover2']:
        r = results[cond]
        print(f"{cond:<12} {r['ssim']:>8.4f}+-{r['ssim_std']:.3f} "
              f"{r['rmse']:>8.4f}+-{r['rmse_std']:.3f} "
              f"{r['mae']:>8.4f}+-{r['mae_std']:.3f} "
              f"{r['psnr']:>8.2f}+-{r['psnr_std']:.2f}")

    # Compute degradation from uncover baseline
    print("\n--- Degradation from Uncover Baseline ---")
    baseline_ssim = results['uncover']['ssim']
    baseline_rmse = results['uncover']['rmse']

    for cond in ['cover1', 'cover2']:
        ssim_drop = (baseline_ssim - results[cond]['ssim']) / baseline_ssim * 100
        rmse_increase = (results[cond]['rmse'] - baseline_rmse) / baseline_rmse * 100
        print(f"{cond}: SSIM drop = {ssim_drop:.1f}%, RMSE increase = {rmse_increase:.1f}%")

    # Save results
    with open(RESULT_DIR / "per_condition_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULT_DIR / 'per_condition_results.json'}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

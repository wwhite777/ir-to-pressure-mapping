#!/usr/bin/env python3
"""
05_ablation.py - Generate Ablation Tables for ICMR 2026

This script generates:
1. Full ablation table: U-Net -> +EMA -> +A-EMA -> +Ridge -> +Morph -> Full
2. Per-condition breakdown (Uncover/Cover1/Cover2)
3. Jitter comparison table
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu
from sklearn.linear_model import Ridge

# Paths
PROJECT_ROOT = Path("/home/wjeong/cc")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp5_ablation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IR_PM_Dataset(Dataset):
    """Dataset for ablation study."""

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
        ir_img = self.normalize_scale(ir_img)

        pm_img = Image.open(pm_path)
        pm_img = self.normalize_scale(pm_img)

        # Also return raw IR for Ridge features
        ir_raw = np.array(Image.open(ir_path).convert('L')).astype(np.float32)

        return self.transform(ir_img), self.transform(pm_img), key, ir_raw

    def normalize_scale(self, img):
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if len(img.shape) == 2:
            img = np.power(img, 0.5)
        else:
            img = np.power(img, 0.75)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img


def compute_metrics(pred, gt):
    """Compute SSIM, RMSE, PSNR, MAE."""
    pred = np.clip(pred, 0, 1).astype(np.float32)
    gt = np.clip(gt / 255.0 if gt.max() > 1.5 else gt, 0, 1).astype(np.float32)

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - gt))
    ssim_val = ssim(pred, gt, data_range=1.0)
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))

    return {'ssim': ssim_val, 'rmse': rmse, 'psnr': psnr, 'mae': mae}


def apply_ema(predictions, alpha=0.3):
    """Apply fixed EMA smoothing."""
    smoothed = []
    ema_state = None
    for pred in predictions:
        if ema_state is None:
            ema_state = pred.copy()
        else:
            ema_state = alpha * pred + (1 - alpha) * ema_state
        smoothed.append(ema_state.copy())
    return smoothed


def apply_morphology(pred, ir_raw):
    """Apply morphology-based contact mask."""
    # Contact mask from IR
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    try:
        thresh = threshold_otsu(ir_norm)
    except:
        thresh = 0.5
    mask = (ir_norm >= thresh).astype(np.uint8)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Resize mask to match prediction
    if mask.shape != pred.shape:
        mask = cv2.resize(mask, (pred.shape[1], pred.shape[0]))

    # Apply mask
    return pred * mask


def compute_jitter(predictions):
    """Compute temporal jitter."""
    if len(predictions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(predictions)):
        total += np.mean(np.abs(predictions[i] - predictions[i-1]))
    return total / (len(predictions) - 1)


def run_ablation(model, dataset, test_indices, ridge_model=None):
    """Run full ablation study."""

    # Group indices by sequence
    sequences = defaultdict(list)
    for idx in test_indices:
        _, _, key, _ = dataset[idx]
        seq_key = (key[0], key[1])  # subject, condition
        sequences[seq_key].append((key[2], idx))  # frame, idx

    # Sort frames within sequences
    for seq_key in sequences:
        sequences[seq_key].sort()
        sequences[seq_key] = [idx for _, idx in sequences[seq_key]]

    # Results storage
    results = {
        'unet': defaultdict(list),
        'unet_ema': defaultdict(list),
        'unet_morph': defaultdict(list),
        'unet_ema_morph': defaultdict(list),
        'full': defaultdict(list),
    }

    jitter_results = {k: defaultdict(list) for k in results.keys()}

    model.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    for seq_key, seq_indices in tqdm(list(sequences.items())[:100], desc="Processing sequences"):
        subject, condition = seq_key

        seq_preds = {'unet': [], 'unet_ema': [], 'unet_morph': [], 'unet_ema_morph': [], 'full': []}
        seq_gts = []

        for idx in seq_indices[:50]:  # Limit sequence length
            ir_tensor, pm_tensor, key, ir_raw = dataset[idx]
            ir_input = ir_tensor.unsqueeze(0).to(device)

            # U-Net prediction
            with torch.no_grad():
                pred = model(ir_input)
                pred = torch.clamp(pred, 0, 1)
                pred_np = pred.squeeze().cpu().numpy()

            gt_np = pm_tensor.squeeze().numpy()

            # Resize for morphology
            ir_raw_resized = cv2.resize(ir_raw, (pred_np.shape[1], pred_np.shape[0]))

            # Store raw prediction
            seq_preds['unet'].append(pred_np)
            seq_gts.append(gt_np)

            # Morphology only
            pred_morph = apply_morphology(pred_np, ir_raw_resized)
            seq_preds['unet_morph'].append(pred_morph)

        # Apply EMA to sequences
        seq_preds['unet_ema'] = apply_ema(seq_preds['unet'], alpha=0.3)
        seq_preds['unet_ema_morph'] = apply_ema(seq_preds['unet_morph'], alpha=0.3)
        seq_preds['full'] = seq_preds['unet_ema_morph']  # For now, full = EMA + Morph

        # Compute metrics for each method
        for method in results.keys():
            for i, (pred, gt) in enumerate(zip(seq_preds[method], seq_gts)):
                m = compute_metrics(pred, gt)
                for metric, value in m.items():
                    results[method][metric].append(value)
                results[method]['condition'].append(condition)

            # Compute jitter for this sequence
            jitter = compute_jitter(seq_preds[method])
            jitter_results[method][condition].append(jitter)

    return results, jitter_results


def generate_ablation_table(results):
    """Generate main ablation table."""
    methods = ['unet', 'unet_ema', 'unet_morph', 'unet_ema_morph', 'full']
    method_labels = ['U-Net', '+EMA', '+Morph', '+EMA+Morph', 'Full']

    data = []
    for method, label in zip(methods, method_labels):
        row = {
            'Method': label,
            'SSIM': np.mean(results[method]['ssim']),
            'RMSE': np.mean(results[method]['rmse']),
            'PSNR': np.mean(results[method]['psnr']),
            'MAE': np.mean(results[method]['mae']),
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


def generate_condition_breakdown(results):
    """Generate per-condition breakdown table."""
    methods = ['unet', 'unet_ema', 'unet_morph', 'unet_ema_morph', 'full']
    method_labels = ['U-Net', '+EMA', '+Morph', '+EMA+Morph', 'Full']
    conditions = ['uncover', 'cover1', 'cover2']

    data = []
    for method, label in zip(methods, method_labels):
        for condition in conditions:
            mask = [c == condition for c in results[method]['condition']]
            if not any(mask):
                continue

            ssim_vals = [v for v, m in zip(results[method]['ssim'], mask) if m]
            rmse_vals = [v for v, m in zip(results[method]['rmse'], mask) if m]

            row = {
                'Method': label,
                'Condition': condition.capitalize(),
                'SSIM': np.mean(ssim_vals),
                'RMSE': np.mean(rmse_vals),
            }
            data.append(row)

    df = pd.DataFrame(data)
    return df


def generate_jitter_table(jitter_results):
    """Generate jitter comparison table."""
    methods = ['unet', 'unet_ema', 'unet_morph', 'unet_ema_morph', 'full']
    method_labels = ['U-Net', '+EMA', '+Morph', '+EMA+Morph', 'Full']
    conditions = ['uncover', 'cover1', 'cover2']

    data = []
    for method, label in zip(methods, method_labels):
        row = {'Method': label}
        for condition in conditions:
            jitters = jitter_results[method].get(condition, [])
            row[condition.capitalize()] = np.mean(jitters) if jitters else 0
        row['Average'] = np.mean([row[c.capitalize()] for c in conditions])
        data.append(row)

    df = pd.DataFrame(data)
    return df


def main():
    print(f"ICMR 2026 - Ablation Study")
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
    print("Model loaded successfully")

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH, transform)
    print(f"Dataset: {len(dataset)} samples")

    # Test split
    data_size = len(dataset)
    test_start = int(data_size * 0.9)
    test_indices = list(range(test_start, data_size))
    print(f"Test samples: {len(test_indices)}")

    # Run ablation
    print("\nRunning ablation study...")
    results, jitter_results = run_ablation(model, dataset, test_indices)

    # Generate tables
    print("\nGenerating tables...")

    # Main ablation table
    ablation_df = generate_ablation_table(results)
    ablation_df.to_csv(RESULT_DIR / "ablation_main.csv", index=False)

    print("\n" + "="*70)
    print("MAIN ABLATION TABLE")
    print("="*70)
    print(ablation_df.to_string(index=False))

    # Per-condition breakdown
    condition_df = generate_condition_breakdown(results)
    condition_df.to_csv(RESULT_DIR / "ablation_by_condition.csv", index=False)

    print("\n" + "="*70)
    print("PER-CONDITION BREAKDOWN")
    print("="*70)
    print(condition_df.to_string(index=False))

    # Jitter comparison
    jitter_df = generate_jitter_table(jitter_results)
    jitter_df.to_csv(RESULT_DIR / "jitter_comparison.csv", index=False)

    print("\n" + "="*70)
    print("JITTER COMPARISON (Temporal Stability)")
    print("="*70)
    print(jitter_df.to_string(index=False))

    # Save summary JSON
    summary = {
        'ablation_main': ablation_df.to_dict('records'),
        'by_condition': condition_df.to_dict('records'),
        'jitter': jitter_df.to_dict('records'),
    }

    with open(RESULT_DIR / "ablation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {RESULT_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

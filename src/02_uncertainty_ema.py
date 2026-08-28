#!/usr/bin/env python3
"""
02_uncertainty_ema.py - Uncertainty-Guided Adaptive EMA for ICMR 2026

This script implements:
1. MC Dropout for uncertainty estimation
2. Adaptive EMA where alpha depends on prediction uncertainty
3. Comparison between fixed EMA and adaptive EMA
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim

# Paths
PROJECT_ROOT = Path(os.environ.get("CC_PROJECT_ROOT",
                                   Path(__file__).resolve().parent.parent))
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp2_uncertainty_ema"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IR_PM_Dataset(Dataset):
    """Dataset for IR to Pressure Map prediction with sequence tracking."""

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

        return self.transform(ir_img), self.transform(pm_img), key

    def normalize_scale(self, img):
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if len(img.shape) == 2:
            img = np.power(img, 0.5)
        else:
            img = np.power(img, 0.75)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img


def enable_dropout(model):
    """Enable dropout layers during evaluation for MC Dropout."""
    enabled_count = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()
            enabled_count += 1
    return enabled_count


def add_dropout_to_decoder(model, p=0.1):
    """
    Add Dropout2d layers after ReLU activations in the decoder.
    This enables MC Dropout for uncertainty estimation.

    The smp.Unet with ResNet50 has no dropout by default.
    We add dropout to decoder blocks to enable uncertainty estimation.
    """
    added_count = 0

    # Add dropout after decoder blocks
    for name, module in model.decoder.named_modules():
        if isinstance(module, nn.Sequential):
            # Find Conv-BN-ReLU sequences and add dropout after ReLU
            new_modules = []
            for child in module.children():
                new_modules.append(child)
                if isinstance(child, nn.ReLU):
                    new_modules.append(nn.Dropout2d(p=p))
                    added_count += 1
            # Replace the sequential with new one including dropout
            if added_count > 0:
                for i, m in enumerate(new_modules):
                    if i < len(list(module.children())):
                        continue  # Skip existing modules
                    module.add_module(f'dropout_{i}', m)

    # Simpler approach: wrap decoder forward with dropout injection
    print(f"Adding MC Dropout to model (p={p})")
    return added_count


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

        # Apply dropout to bottleneck features (deepest layer)
        # features is a list of feature maps at different scales
        features_with_dropout = []
        for i, f in enumerate(features):
            if i >= len(features) - 2:  # Apply dropout to deepest 2 layers
                f = self.dropout(f)
            features_with_dropout.append(f)

        # Decode with dropout-applied features (pass as list, not unpacked)
        decoder_output = self.model.decoder(features_with_dropout)

        # Apply dropout before segmentation head
        decoder_output = self.dropout(decoder_output)

        # Final segmentation
        masks = self.model.segmentation_head(decoder_output)
        return masks

    def encoder(self, x):
        return self.model.encoder(x)


def mc_dropout_predict(model, x, n_samples=10):
    """
    Perform MC Dropout inference to get mean prediction and uncertainty.

    Returns:
        mean_pred: Mean prediction across samples
        uncertainty: Pixel-wise uncertainty (std across samples)
    """
    model.eval()
    enable_dropout(model)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)  # [n_samples, B, C, H, W]
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)

    return mean_pred, uncertainty


def fixed_ema(predictions, alpha=0.3):
    """Apply fixed EMA smoothing to a sequence of predictions."""
    smoothed = []
    ema_state = None

    for pred in predictions:
        if ema_state is None:
            ema_state = pred.copy()
        else:
            ema_state = alpha * pred + (1 - alpha) * ema_state
        smoothed.append(ema_state.copy())

    return smoothed


def adaptive_ema(predictions, uncertainties, alpha_min=0.1, alpha_max=0.5, verbose=False):
    """
    Apply uncertainty-guided adaptive EMA smoothing.

    High uncertainty -> stronger smoothing (lower alpha, trust history more)
    Low uncertainty -> weaker smoothing (higher alpha, trust current more)

    Formula: α_t = α_max - (u_t - u_min)/(u_max - u_min) * (α_max - α_min)
    """
    smoothed = []
    ema_state = None
    alphas = []

    # Compute mean uncertainty per frame
    mean_uncertainties = np.array([u.mean() for u in uncertainties])

    # Normalize uncertainties to [0, 1] range for adaptive alpha
    u_min = mean_uncertainties.min()
    u_max = mean_uncertainties.max()
    u_range = u_max - u_min + 1e-8

    if verbose:
        print(f"  Uncertainty stats: min={u_min:.6f}, max={u_max:.6f}, range={u_range:.6f}")

    for pred, unc in zip(predictions, uncertainties):
        # Normalize uncertainty
        u_mean = unc.mean()
        u_norm = (u_mean - u_min) / u_range

        # High uncertainty -> low alpha (trust history)
        # Low uncertainty -> high alpha (trust current)
        alpha = alpha_max - u_norm * (alpha_max - alpha_min)
        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        alphas.append(alpha)

        if ema_state is None:
            ema_state = pred.copy()
        else:
            ema_state = alpha * pred + (1 - alpha) * ema_state
        smoothed.append(ema_state.copy())

    if verbose:
        alphas_arr = np.array(alphas)
        print(f"  Alpha stats: min={alphas_arr.min():.4f}, max={alphas_arr.max():.4f}, mean={alphas_arr.mean():.4f}")
        # Check correlation
        corr = np.corrcoef(mean_uncertainties, alphas)[0, 1]
        print(f"  Correlation(u, α) = {corr:.4f} (should be negative)")
        if corr > 0:
            print("  WARNING: Positive correlation! α logic may be inverted!")

    return smoothed, alphas


def validate_alpha_logic(uncertainties, alphas):
    """
    Validate that α(u) behaves correctly.
    Returns True if logic is correct (negative correlation).
    """
    mean_uncertainties = np.array([u.mean() for u in uncertainties])
    alphas_arr = np.array(alphas)

    print(f"\n--- Alpha Logic Validation ---")
    print(f"Uncertainty: min={mean_uncertainties.min():.6f}, max={mean_uncertainties.max():.6f}, "
          f"mean={mean_uncertainties.mean():.6f}, std={mean_uncertainties.std():.6f}")
    print(f"Alpha: min={alphas_arr.min():.4f}, max={alphas_arr.max():.4f}, "
          f"mean={alphas_arr.mean():.4f}, std={alphas_arr.std():.4f}")

    corr = np.corrcoef(mean_uncertainties, alphas_arr)[0, 1]
    print(f"Correlation(u, α) = {corr:.4f}")

    if np.isnan(corr):
        print("WARNING: Correlation is NaN - uncertainty may be constant!")
        return False
    elif corr > 0:
        print("WARNING: Positive correlation - α logic is INVERTED!")
        return False
    else:
        print("OK: Negative correlation - α logic is correct")
        return True


def compute_jitter(predictions):
    """Compute temporal jitter (mean absolute difference between consecutive frames)."""
    if len(predictions) < 2:
        return 0.0

    total_jitter = 0.0
    for i in range(1, len(predictions)):
        diff = np.abs(predictions[i] - predictions[i-1])
        total_jitter += diff.mean()

    return total_jitter / (len(predictions) - 1)


def compute_metrics(pred, gt):
    """Compute SSIM, RMSE, MAE for a single prediction."""
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt / 255.0 if gt.max() > 1.5 else gt, 0, 1)

    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - gt))
    ssim_val = ssim(pred, gt, data_range=1.0)

    return {'ssim': ssim_val, 'rmse': rmse, 'mae': mae}


def process_sequence(model, dataset, indices, n_mc_samples=20, verbose=False):
    """
    Process a sequence of frames and compare fixed vs adaptive EMA.

    Returns metrics for: raw, fixed_ema, adaptive_ema
    """
    model.eval()
    # Ensure dropout is enabled for MC Dropout
    enable_dropout(model)

    # Collect predictions for the sequence
    raw_preds = []
    uncertainties = []
    gts = []

    for idx in indices:
        ir, pm, key = dataset[idx]
        ir = ir.unsqueeze(0).to(device)

        # MC Dropout prediction
        mean_pred, unc = mc_dropout_predict(model, ir, n_samples=n_mc_samples)

        raw_preds.append(mean_pred.squeeze().cpu().numpy())
        uncertainties.append(unc.squeeze().cpu().numpy())
        gts.append(pm.squeeze().numpy())

    # Apply smoothing methods
    fixed_smoothed = fixed_ema(raw_preds, alpha=0.3)
    adaptive_smoothed, alphas = adaptive_ema(raw_preds, uncertainties, verbose=verbose)

    # Compute metrics
    results = {
        'raw': {'ssim': [], 'rmse': [], 'mae': []},
        'fixed_ema': {'ssim': [], 'rmse': [], 'mae': []},
        'adaptive_ema': {'ssim': [], 'rmse': [], 'mae': []},
    }

    for i, gt in enumerate(gts):
        for method, preds in [('raw', raw_preds), ('fixed_ema', fixed_smoothed), ('adaptive_ema', adaptive_smoothed)]:
            m = compute_metrics(preds[i], gt)
            for k, v in m.items():
                results[method][k].append(v)

    # Compute jitter
    results['raw']['jitter'] = compute_jitter(raw_preds)
    results['fixed_ema']['jitter'] = compute_jitter(fixed_smoothed)
    results['adaptive_ema']['jitter'] = compute_jitter(adaptive_smoothed)

    # Average metrics
    for method in results:
        for metric in ['ssim', 'rmse', 'mae']:
            results[method][metric] = np.mean(results[method][metric])

    results['alphas'] = alphas
    results['mean_uncertainty'] = [u.mean() for u in uncertainties]

    return results


def main():
    print(f"ICMR 2026 - Uncertainty-Guided Adaptive EMA")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Load model
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    base_model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    # Filter out thop profiling keys
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    base_model.load_state_dict(state_dict, strict=False)

    # Wrap with MC Dropout for uncertainty estimation
    model = MCDropoutWrapper(base_model, p=0.1)
    model = model.to(device)
    print("Model loaded successfully")

    # Verify dropout is now present
    dropout_count = sum(1 for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d)))
    print(f"MC Dropout layers: {dropout_count}")

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH, transform)
    print(f"Dataset: {len(dataset)} samples")

    # Group by subject and condition to process sequences
    sequences = {}
    for idx, (_, _, key) in enumerate(dataset.pairs):
        subject, condition, frame = key
        seq_key = (subject, condition)
        if seq_key not in sequences:
            sequences[seq_key] = []
        sequences[seq_key].append((frame, idx))

    # Sort frames within each sequence
    for seq_key in sequences:
        sequences[seq_key].sort(key=lambda x: x[0])
        sequences[seq_key] = [idx for _, idx in sequences[seq_key]]

    print(f"Found {len(sequences)} sequences")

    # Process a subset of sequences for analysis
    all_results = []
    conditions_to_analyze = ['uncover', 'cover1', 'cover2']

    first_seq = True
    for (subject, condition), indices in tqdm(list(sequences.items())[:50], desc="Processing sequences"):
        if condition not in conditions_to_analyze:
            continue
        if len(indices) < 10:  # Skip very short sequences
            continue

        # Verbose for first sequence to validate alpha logic
        results = process_sequence(model, dataset, indices[:50], n_mc_samples=20, verbose=first_seq)
        results['subject'] = subject
        results['condition'] = condition
        all_results.append(results)

        # Validate alpha logic on first sequence
        if first_seq and 'alphas' in results and 'mean_uncertainty' in results:
            # Convert mean_uncertainty to array format expected by validate_alpha_logic
            unc_arrays = [np.array([u]) for u in results['mean_uncertainty']]
            validate_alpha_logic(unc_arrays, results['alphas'])
            first_seq = False

    # Aggregate results
    summary = {
        'raw': {'ssim': [], 'rmse': [], 'mae': [], 'jitter': []},
        'fixed_ema': {'ssim': [], 'rmse': [], 'mae': [], 'jitter': []},
        'adaptive_ema': {'ssim': [], 'rmse': [], 'mae': [], 'jitter': []},
    }

    for r in all_results:
        for method in ['raw', 'fixed_ema', 'adaptive_ema']:
            for metric in ['ssim', 'rmse', 'mae', 'jitter']:
                summary[method][metric].append(r[method][metric])

    # Print summary
    print(f"\n{'='*70}")
    print("UNCERTAINTY-GUIDED ADAPTIVE EMA RESULTS")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'SSIM':>8} {'RMSE':>8} {'MAE':>8} {'Jitter':>10}")
    print("-" * 70)

    for method in ['raw', 'fixed_ema', 'adaptive_ema']:
        s = summary[method]
        print(f"{method:<15} {np.mean(s['ssim']):>8.4f} {np.mean(s['rmse']):>8.4f} "
              f"{np.mean(s['mae']):>8.4f} {np.mean(s['jitter']):>10.6f}")

    # Save results
    results_df = pd.DataFrame([
        {'method': method, 'metric': metric, 'value': np.mean(summary[method][metric])}
        for method in summary
        for metric in summary[method]
    ])
    results_df.to_csv(RESULT_DIR / "ema_comparison.csv", index=False)

    # Save detailed results
    with open(RESULT_DIR / "detailed_results.json", 'w') as f:
        json.dump({
            'summary': {m: {k: float(np.mean(v)) for k, v in s.items()} for m, s in summary.items()},
            'jitter_reduction_fixed': float(1 - np.mean(summary['fixed_ema']['jitter']) / np.mean(summary['raw']['jitter'])),
            'jitter_reduction_adaptive': float(1 - np.mean(summary['adaptive_ema']['jitter']) / np.mean(summary['raw']['jitter'])),
        }, f, indent=2)

    print(f"\nResults saved to: {RESULT_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

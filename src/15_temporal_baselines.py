#!/usr/bin/env python3
"""
15_temporal_baselines.py - Temporal Filtering Baselines

Compares UA-EMA against:
1. No filtering (raw)
2. Moving average
3. Fixed EMA
4. Kalman filter
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
from torch.utils.data import Dataset
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
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp7_temporal"
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


def enable_dropout(model):
    """Enable dropout layers for MC Dropout."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def mc_dropout_predict(model, x, n_samples=10):
    """MC Dropout prediction."""
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


# Temporal filtering methods
def no_filter(predictions, **kwargs):
    """No filtering - return raw predictions."""
    return predictions


def moving_average(predictions, k=5, **kwargs):
    """Simple moving average filter."""
    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - k + 1)
        smoothed.append(np.mean(predictions[start:i+1], axis=0))
    return smoothed


def fixed_ema(predictions, alpha=0.3, **kwargs):
    """Fixed exponential moving average."""
    smoothed = []
    state = None
    for pred in predictions:
        if state is None:
            state = pred.copy()
        else:
            state = alpha * pred + (1 - alpha) * state
        smoothed.append(state.copy())
    return smoothed


def adaptive_ema(predictions, uncertainties, alpha_min=0.1, alpha_max=0.5, **kwargs):
    """Uncertainty-guided adaptive EMA."""
    smoothed = []
    state = None

    mean_uncs = np.array([u.mean() for u in uncertainties])
    u_min, u_max = mean_uncs.min(), mean_uncs.max()
    u_range = u_max - u_min + 1e-8

    for pred, unc in zip(predictions, uncertainties):
        u_norm = (unc.mean() - u_min) / u_range
        alpha = alpha_max - u_norm * (alpha_max - alpha_min)
        alpha = np.clip(alpha, alpha_min, alpha_max)

        if state is None:
            state = pred.copy()
        else:
            state = alpha * pred + (1 - alpha) * state
        smoothed.append(state.copy())

    return smoothed


def kalman_filter(predictions, uncertainties=None, process_noise=0.01, **kwargs):
    """
    Simple Kalman filter for temporal smoothing.

    State: x_t (current pressure)
    Measurement: z_t (prediction)

    x_t = x_{t-1}  (constant state model)
    z_t = x_t + v_t  (measurement with noise)

    If uncertainties provided, use them as measurement noise R.
    """
    smoothed = []
    x = None  # State estimate
    P = 1.0  # Error covariance

    Q = process_noise  # Process noise

    for i, pred in enumerate(predictions):
        if x is None:
            x = pred.copy()
            P = 1.0
        else:
            # Predict step
            x_pred = x  # State doesn't change
            P_pred = P + Q

            # Update step
            if uncertainties is not None:
                R = float(uncertainties[i].mean()) + 0.001  # Measurement noise from uncertainty
            else:
                R = 0.1  # Default measurement noise

            K = P_pred / (P_pred + R)  # Kalman gain
            x = x_pred + K * (pred - x_pred)
            P = (1 - K) * P_pred

        smoothed.append(x.copy())

    return smoothed


def compute_jitter(predictions):
    """Compute temporal jitter."""
    if len(predictions) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(predictions)):
        total += np.abs(predictions[i] - predictions[i-1]).mean()
    return total / (len(predictions) - 1)


def compute_ssim(preds, gts):
    """Compute average SSIM."""
    ssims = []
    for pred, gt in zip(preds, gts):
        pred = np.clip(pred, 0, 1)
        gt = np.clip(gt, 0, 1)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        ssims.append(ssim(pred, gt, data_range=1.0))
    return np.mean(ssims)


def main():
    print(f"ICMR 2026 - Temporal Filtering Baselines")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Load model with MC Dropout
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    base_model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    base_model.load_state_dict(state_dict, strict=False)

    model = MCDropoutWrapper(base_model, p=0.1)
    model = model.to(device)
    print("Model loaded with MC Dropout")

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH, transform)
    print(f"Dataset: {len(dataset)} samples")

    # Group by sequence (subject, condition)
    sequences = {}
    for idx, (_, _, key) in enumerate(dataset.pairs):
        seq_key = (key[0], key[1])
        if seq_key not in sequences:
            sequences[seq_key] = []
        sequences[seq_key].append((key[2], idx))

    # Sort frames within each sequence
    for seq_key in sequences:
        sequences[seq_key].sort(key=lambda x: x[0])
        sequences[seq_key] = [idx for _, idx in sequences[seq_key]]

    print(f"Found {len(sequences)} sequences")

    # Filter methods to compare
    methods = {
        'No Filter': (no_filter, {}),
        'Moving Avg (k=5)': (moving_average, {'k': 5}),
        'Fixed EMA (α=0.3)': (fixed_ema, {'alpha': 0.3}),
        'Kalman Filter': (kalman_filter, {}),
        'Kalman + Unc': (kalman_filter, {}),  # Will pass uncertainties
        'Adaptive EMA (Ours)': (adaptive_ema, {}),
    }

    # Process sequences
    results = {method: {'ssim': [], 'jitter': []} for method in methods}

    n_sequences = 30  # Number of sequences to evaluate
    seq_keys = list(sequences.keys())[:n_sequences]

    for seq_key in tqdm(seq_keys, desc="Sequences"):
        indices = sequences[seq_key][:50]  # Limit sequence length
        if len(indices) < 10:
            continue

        # Get predictions and uncertainties
        raw_preds = []
        uncertainties = []
        gts = []

        for idx in indices:
            ir, pm, _ = dataset[idx]
            ir = ir.unsqueeze(0).to(device)

            mean_pred, unc = mc_dropout_predict(model, ir, n_samples=10)
            pred = mean_pred.squeeze().cpu().numpy()
            gt = pm.squeeze().numpy()

            # Resize
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            unc_np = unc.squeeze().cpu().numpy()
            unc_np = cv2.resize(unc_np, (gt.shape[1], gt.shape[0]))

            raw_preds.append(pred)
            uncertainties.append(unc_np)
            gts.append(gt)

        # Apply each filtering method
        for method_name, (func, params) in methods.items():
            if 'Unc' in method_name or 'Adaptive' in method_name:
                smoothed = func(raw_preds, uncertainties=uncertainties, **params)
            else:
                smoothed = func(raw_preds, **params)

            ssim_val = compute_ssim(smoothed, gts)
            jitter = compute_jitter(smoothed)

            results[method_name]['ssim'].append(ssim_val)
            results[method_name]['jitter'].append(jitter)

    # Aggregate results
    print(f"\n{'='*80}")
    print("TEMPORAL FILTERING BASELINES")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'SSIM':>10} {'Jitter':>12} {'Jitter Red.':>12}")
    print("-" * 80)

    raw_jitter = np.mean(results['No Filter']['jitter'])

    summary = {}
    for method_name in methods:
        ssim_mean = np.mean(results[method_name]['ssim'])
        jitter_mean = np.mean(results[method_name]['jitter'])
        jitter_reduction = (1 - jitter_mean / raw_jitter) * 100 if raw_jitter > 0 else 0

        summary[method_name] = {
            'ssim': float(ssim_mean),
            'jitter': float(jitter_mean),
            'jitter_reduction': float(jitter_reduction)
        }

        print(f"{method_name:<25} {ssim_mean:>10.4f} {jitter_mean:>12.6f} {jitter_reduction:>10.1f}%")

    # Save results
    with open(RESULT_DIR / "temporal_baselines.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {RESULT_DIR / 'temporal_baselines.json'}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

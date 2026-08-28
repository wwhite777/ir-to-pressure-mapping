#!/usr/bin/env python3
"""
01_train_baselines.py - Train U-Net++ and DeepLabV3+ baselines for ICMR 2026

This script trains two additional network architectures alongside the existing U-Net:
- U-Net++ (UnetPlusPlus) with ResNet50 encoder
- DeepLabV3+ with ResNet50 encoder

All models are trained on the SLP IR-to-Pressure mapping task.
"""

import os
import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm

# Paths
PROJECT_ROOT = Path(os.environ.get("CC_PROJECT_ROOT",
                                   Path(__file__).resolve().parent.parent))
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp1_baselines"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class IR_PM_Dataset(Dataset):
    """Dataset for IR to Pressure Map prediction."""

    def __init__(self, ir_folder, pm_folder, transform):
        self.ir_folder = ir_folder
        self.pm_folder = pm_folder
        self.transform = transform

        import re
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

        print(f"Dataset: {len(self.pairs)} pairs loaded")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ir_path, pm_path, key = self.pairs[idx]

        ir_img = Image.open(ir_path).convert('RGB')
        ir_img = self.normalize_scale(ir_img)

        pm_img = Image.open(pm_path)
        pm_img = self.normalize_scale(pm_img)

        ir_tensor = self.transform(ir_img)
        pm_tensor = self.transform(pm_img)

        return ir_tensor, pm_tensor, key

    def normalize_scale(self, img):
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if len(img.shape) == 2:
            img = np.power(img, 0.5)
        else:
            img = np.power(img, 0.75)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img


def get_model(model_name: str, encoder_name: str = "resnet50"):
    """Create a segmentation model."""
    if model_name == "unet":
        model = smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet",
                         in_channels=3, classes=1)
    elif model_name == "unetplusplus":
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights="imagenet",
                                  in_channels=3, classes=1)
    elif model_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights="imagenet",
                                   in_channels=3, classes=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def RMSE(y_pred, y_true):
    return torch.sqrt(F.mse_loss(y_pred, y_true))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_rmse = 0.0

    for ir, pm, _ in tqdm(loader, desc="Training", leave=False):
        ir, pm = ir.to(device), pm.to(device)

        optimizer.zero_grad()
        pred = model(ir)
        loss = criterion(pred, pm)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rmse += RMSE(pred, pm).item()

    n = len(loader)
    return total_loss / n, total_rmse / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_mae = 0.0

    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for ir, pm, _ in tqdm(loader, desc="Evaluating", leave=False):
            ir, pm = ir.to(device), pm.to(device)

            pred = model(ir)
            pred = torch.clamp(pred, 0.0, 1.0)
            pm = torch.clamp(pm, 0.0, 1.0)

            loss = criterion(pred, pm)
            total_loss += loss.item()

            rmse = RMSE(pred, pm)
            total_rmse += rmse.item()

            ssim = ssim_metric(pred, pm)
            total_ssim += ssim.item()

            mae = torch.mean(torch.abs(pred - pm))
            total_mae += mae.item()

            mse = F.mse_loss(pred, pm).item()
            psnr = 10.0 * math.log10(1.0 / (mse + 1e-12))
            total_psnr += psnr

    n = len(loader)
    return {
        'loss': total_loss / n,
        'rmse': total_rmse / n,
        'ssim': total_ssim / n,
        'psnr': total_psnr / n,
        'mae': total_mae / n
    }


def train_model(model_name: str, num_epochs: int = 50, batch_size: int = 32):
    """Train a single model and save results."""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((192, 96))
    ])
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH, transform)

    # Split
    data_size = len(dataset)
    train_size = int(data_size * 0.8)
    val_size = int(data_size * 0.1)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, data_size))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = get_model(model_name).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_ssim': []}

    for epoch in tqdm(range(num_epochs), desc=f"Training {model_name}"):
        t0 = time.time()

        train_loss, train_rmse = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_ssim'].append(val_metrics['ssim'])

        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) - "
                  f"Train Loss: {train_loss:.6f}, Val RMSE: {val_metrics['rmse']:.4f}, "
                  f"Val SSIM: {val_metrics['ssim']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = MODEL_DIR / f"{model_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)

    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  SSIM: {test_metrics['ssim']:.4f}")
    print(f"  PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"  MAE:  {test_metrics['mae']:.4f}")

    # Save final model
    final_path = MODEL_DIR / f"{model_name}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'history': history,
    }, final_path)

    # Save predictions for test set
    print("\nSaving predictions...")
    pred_dir = RESULT_DIR / f"{model_name}_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, (ir, pm, keys) in enumerate(tqdm(test_loader, desc="Saving preds")):
            ir = ir.to(device)
            pred = model(ir).cpu().numpy()

            # keys is a tuple of batched values: (subjects, conditions, frames)
            batch_size = len(keys[0])
            for j in range(batch_size):
                subject = keys[0][j].item() if hasattr(keys[0][j], 'item') else keys[0][j]
                condition = keys[1][j]
                frame = keys[2][j].item() if hasattr(keys[2][j], 'item') else keys[2][j]
                stem = f"{subject:05d}_ir_{condition}_image_{frame:06d}"
                np.save(pred_dir / f"{stem}.npy", pred[j].squeeze().astype(np.float32))

    return test_metrics, history


def load_and_save_predictions(model_name):
    """Load a trained model and save predictions for test set."""
    final_path = MODEL_DIR / f"{model_name}_final.pth"
    if not final_path.exists():
        return None

    print(f"\nLoading saved model: {model_name}")

    # Create model
    if model_name == 'unetplusplus':
        model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=None,
                                  in_channels=3, classes=1)
    elif model_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None,
                                   in_channels=3, classes=1)
    else:
        return None

    checkpoint = torch.load(final_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Get test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH, transform)

    n = len(dataset)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)
    test_indices = list(range(train_n + val_n, n))
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    # Save predictions
    print("Saving predictions...")
    pred_dir = RESULT_DIR / f"{model_name}_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, (ir, pm, keys) in enumerate(tqdm(test_loader, desc="Saving preds")):
            ir = ir.to(device)
            pred = model(ir).cpu().numpy()

            batch_size = len(keys[0])
            for j in range(batch_size):
                subject = keys[0][j].item() if hasattr(keys[0][j], 'item') else keys[0][j]
                condition = keys[1][j]
                frame = keys[2][j].item() if hasattr(keys[2][j], 'item') else keys[2][j]
                stem = f"{subject:05d}_ir_{condition}_image_{frame:06d}"
                np.save(pred_dir / f"{stem}.npy", pred[j].squeeze().astype(np.float32))

    return checkpoint.get('test_metrics', {})


def main():
    print(f"ICMR 2026 - Baseline Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    results = {}

    # Check if U-Net++ already trained
    unetpp_final = MODEL_DIR / "unetplusplus_final.pth"
    if unetpp_final.exists():
        print(f"\nU-Net++ already trained, loading and saving predictions...")
        saved_metrics = load_and_save_predictions('unetplusplus')
        if saved_metrics:
            results['unetplusplus'] = saved_metrics
        else:
            # Re-evaluate if metrics not saved
            results['unetplusplus'], _ = train_model('unetplusplus', num_epochs=50)
    else:
        results['unetplusplus'], _ = train_model('unetplusplus', num_epochs=50)

    # Check if DeepLabV3+ already trained
    deeplabv3_final = MODEL_DIR / "deeplabv3plus_final.pth"
    if deeplabv3_final.exists():
        print(f"\nDeepLabV3+ already trained, loading and saving predictions...")
        saved_metrics = load_and_save_predictions('deeplabv3plus')
        if saved_metrics:
            results['deeplabv3plus'] = saved_metrics
        else:
            results['deeplabv3plus'], _ = train_model('deeplabv3plus', num_epochs=50)
    else:
        results['deeplabv3plus'], _ = train_model('deeplabv3plus', num_epochs=50)

    # Save summary
    summary_path = RESULT_DIR / "baseline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'RMSE':>8} {'SSIM':>8} {'PSNR':>8} {'MAE':>8}")
    print("-" * 60)

    # Load existing U-Net results for comparison
    unet_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    if unet_path.exists():
        print(f"{'U-Net':<15} {'(existing model - evaluate separately)':>40}")

    for name, metrics in results.items():
        print(f"{name:<15} {metrics['rmse']:>8.4f} {metrics['ssim']:>8.4f} "
              f"{metrics['psnr']:>8.2f} {metrics['mae']:>8.4f}")

    print(f"\nResults saved to: {RESULT_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

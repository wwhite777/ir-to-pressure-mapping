#!/usr/bin/env python3
"""
14_tsne_visualization.py - t-SNE/UMAP Visualization of Embedding Space

Creates visualization showing that:
1. Embeddings cluster by pressure patterns (not just subject identity)
2. Different occlusion conditions intermix if embeddings are occlusion-invariant
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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2

# Paths
PROJECT_ROOT = Path("/home/wjeong/cc")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp5_tsne"
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


def main():
    print(f"ICMR 2026 - t-SNE Visualization")
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

    # Sample subset for visualization (balanced by condition)
    condition_samples = {'uncover': [], 'cover1': [], 'cover2': []}
    for idx, (_, _, key) in enumerate(dataset.pairs):
        cond = key[1]
        if cond in condition_samples:
            condition_samples[cond].append(idx)

    # Balance sampling
    n_per_condition = 300
    np.random.seed(42)
    sampled_indices = []
    for cond in condition_samples:
        indices = condition_samples[cond]
        np.random.shuffle(indices)
        sampled_indices.extend(indices[:n_per_condition])

    print(f"Sampling {len(sampled_indices)} samples for t-SNE")

    # Extract embeddings
    embeddings = []
    conditions = []
    subjects = []
    mean_pressures = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for idx in tqdm(sampled_indices, desc="Extracting"):
            ir, pm, key = dataset[idx]
            ir = ir.unsqueeze(0).to(device)

            # Get encoder features (bottleneck)
            encoder_features = model.encoder(ir)
            bottleneck = encoder_features[-1]  # Deepest features

            # Global average pooling
            embedding = bottleneck.mean(dim=[2, 3]).flatten().cpu().numpy()
            embeddings.append(embedding)

            # Metadata
            conditions.append(key[1])
            subjects.append(key[0])
            mean_pressures.append(pm.mean().item())

    embeddings = np.vstack(embeddings)
    conditions = np.array(conditions)
    subjects = np.array(subjects)
    mean_pressures = np.array(mean_pressures)

    print(f"Embedding shape: {embeddings.shape}")

    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)

    # Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Map conditions to colors
    condition_colors = {'uncover': 'blue', 'cover1': 'orange', 'cover2': 'green'}
    condition_labels = {'uncover': 'Uncover', 'cover1': 'Cover1', 'cover2': 'Cover2'}
    colors = [condition_colors[c] for c in conditions]

    # Panel (a): Color by condition
    for cond in ['uncover', 'cover1', 'cover2']:
        mask = conditions == cond
        axes[0].scatter(tsne_proj[mask, 0], tsne_proj[mask, 1],
                       c=condition_colors[cond], label=condition_labels[cond],
                       alpha=0.6, s=20)
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[0].set_title('(a) Colored by Occlusion Condition', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel (b): Color by pressure intensity
    scatter = axes[1].scatter(tsne_proj[:, 0], tsne_proj[:, 1],
                              c=mean_pressures, cmap='jet', alpha=0.6, s=20)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=11)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=11)
    axes[1].set_title('(b) Colored by Mean Pressure Intensity', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Mean Pressure', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('t-SNE Visualization of IR Encoder Embeddings', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = RESULT_DIR / "tsne_embeddings.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

    # Additional analysis: cluster overlap
    print("\n--- Cluster Analysis ---")

    # Check if conditions overlap (good for occlusion-invariance)
    # Compute centroid of each condition
    for cond in ['uncover', 'cover1', 'cover2']:
        mask = conditions == cond
        centroid = tsne_proj[mask].mean(axis=0)
        spread = tsne_proj[mask].std(axis=0).mean()
        print(f"{cond}: centroid=({centroid[0]:.2f}, {centroid[1]:.2f}), spread={spread:.2f}")

    # Compute pairwise centroid distances
    centroids = {}
    for cond in ['uncover', 'cover1', 'cover2']:
        mask = conditions == cond
        centroids[cond] = tsne_proj[mask].mean(axis=0)

    print("\nCentroid distances (lower = more overlap = more invariant):")
    for c1 in ['uncover', 'cover1', 'cover2']:
        for c2 in ['uncover', 'cover1', 'cover2']:
            if c1 < c2:
                dist = np.linalg.norm(centroids[c1] - centroids[c2])
                print(f"  {c1} <-> {c2}: {dist:.2f}")

    # Save metadata
    metadata = {
        'n_samples': len(sampled_indices),
        'n_per_condition': n_per_condition,
        'conditions': list(np.unique(conditions)),
        'centroids': {k: v.tolist() for k, v in centroids.items()},
        'mean_pressure_range': [float(mean_pressures.min()), float(mean_pressures.max())]
    }
    with open(RESULT_DIR / "tsne_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
03_retrieval.py - Cross-Modal Retrieval Experiment for ICMR 2026

This script implements:
1. Extract encoder embeddings from U-Net
2. Define retrieval task based on pressure-pattern similarity
3. Compute Recall@K (K=1,5,10) and mAP
4. Generate retrieval visualization
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Paths
PROJECT_ROOT = Path("/home/wjeong/cc")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "exp3_retrieval"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IR_PM_Dataset(Dataset):
    """Dataset for IR to Pressure Map prediction."""

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

        return self.transform(ir_img), self.transform(pm_img), key, ir_path, pm_path

    def normalize_scale(self, img):
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if len(img.shape) == 2:
            img = np.power(img, 0.5)
        else:
            img = np.power(img, 0.75)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img


class EncoderExtractor(nn.Module):
    """Extract encoder features from U-Net."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, x):
        features = self.encoder(x)
        # Use the deepest feature map (bottleneck)
        bottleneck = features[-1]
        # Global average pooling
        embedding = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        return embedding


def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all samples in the dataloader."""
    model.eval()
    embeddings = []
    keys = []
    ir_paths = []
    pm_paths = []

    with torch.no_grad():
        for ir, pm, key, ir_path, pm_path in tqdm(dataloader, desc="Extracting embeddings"):
            ir = ir.to(device)
            emb = model(ir)
            embeddings.append(emb.cpu().numpy())

            # Handle batch
            for i in range(len(key[0])):
                k = (key[0][i].item(), key[1][i], key[2][i].item())
                keys.append(k)
                ir_paths.append(ir_path[i])
                pm_paths.append(pm_path[i])

    embeddings = np.vstack(embeddings)
    return embeddings, keys, ir_paths, pm_paths


def compute_pressure_similarity(pm_paths, sample_indices=None):
    """
    Compute pairwise pressure map similarity matrix.
    Uses downsampled pressure maps for efficiency.
    """
    if sample_indices is not None:
        pm_paths = [pm_paths[i] for i in sample_indices]

    n = len(pm_paths)
    pressure_maps = []

    for path in tqdm(pm_paths, desc="Loading pressure maps"):
        pm = np.array(Image.open(path).convert('L'))
        # Downsample for efficiency
        pm = pm[::4, ::4].astype(np.float32) / 255.0
        pressure_maps.append(pm.flatten())

    pressure_maps = np.array(pressure_maps)

    # Compute cosine similarity
    similarity = cosine_similarity(pressure_maps)
    return similarity


def compute_recall_at_k(embeddings, similarity_matrix, k_values=[1, 5, 10], threshold=0.8):
    """
    Compute Recall@K for retrieval task.

    A retrieval is considered correct if the ground truth pressure similarity
    is above the threshold.
    """
    n = len(embeddings)

    # Find nearest neighbors in embedding space
    nn = NearestNeighbors(n_neighbors=max(k_values) + 1, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    recalls = {k: [] for k in k_values}

    for i in range(n):
        # Get retrieved indices (excluding self)
        retrieved = indices[i, 1:max(k_values)+1]

        # Check if any retrieved sample has high pressure similarity
        for k in k_values:
            top_k = retrieved[:k]
            # A match is when pressure similarity > threshold
            matches = [similarity_matrix[i, j] > threshold for j in top_k]
            recalls[k].append(1.0 if any(matches) else 0.0)

    return {k: np.mean(v) for k, v in recalls.items()}


def compute_map(embeddings, similarity_matrix, threshold=0.8):
    """Compute mean Average Precision for retrieval."""
    n = len(embeddings)

    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=n, metric='cosine')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    aps = []

    for i in range(n):
        retrieved = indices[i, 1:]  # Exclude self

        # Compute precision at each rank where we have a relevant item
        precisions = []
        num_relevant = 0

        for rank, j in enumerate(retrieved, 1):
            if similarity_matrix[i, j] > threshold:
                num_relevant += 1
                precisions.append(num_relevant / rank)

        if precisions:
            aps.append(np.mean(precisions))
        else:
            aps.append(0.0)

    return np.mean(aps)


def create_retrieval_visualization(embeddings, keys, ir_paths, pm_paths, similarity_matrix,
                                   query_indices, k=5, save_path=None):
    """Create visualization showing query and top-K retrieved results."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn.fit(embeddings)

    fig, axes = plt.subplots(len(query_indices), k + 1, figsize=(3 * (k + 1), 3 * len(query_indices)))

    for row, q_idx in enumerate(query_indices):
        distances, indices = nn.kneighbors(embeddings[q_idx:q_idx+1])
        retrieved = indices[0, 1:k+1]

        # Query
        ax = axes[row, 0] if len(query_indices) > 1 else axes[0]
        ir = np.array(Image.open(ir_paths[q_idx]))
        ax.imshow(ir)
        subject, condition, frame = keys[q_idx]
        ax.set_title(f"Query\nS{subject}/{condition}", fontsize=8)
        ax.axis('off')

        # Retrieved
        for col, r_idx in enumerate(retrieved):
            ax = axes[row, col + 1] if len(query_indices) > 1 else axes[col + 1]
            ir = np.array(Image.open(ir_paths[r_idx]))
            ax.imshow(ir)
            subject, condition, frame = keys[r_idx]
            sim = similarity_matrix[q_idx, r_idx]
            ax.set_title(f"S{subject}/{condition}\nSim: {sim:.2f}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()


def main():
    print(f"ICMR 2026 - Cross-Modal Retrieval Experiment")
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

    encoder = EncoderExtractor(base_model).to(device)
    print("Model loaded successfully")

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH, transform)

    # Use test split
    data_size = len(dataset)
    test_start = int(data_size * 0.9)
    test_indices = list(range(test_start, data_size))

    # For efficiency, sample a subset
    sample_size = min(1000, len(test_indices))
    np.random.seed(42)
    sample_indices = np.random.choice(test_indices, sample_size, replace=False)

    # Create subset dataset
    class SubsetDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    subset = SubsetDataset(dataset, sample_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings, keys, ir_paths, pm_paths = extract_embeddings(encoder, loader, device)
    print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Compute pressure similarity matrix
    print("\nComputing pressure similarity matrix...")
    similarity_matrix = compute_pressure_similarity(pm_paths)

    # Compute retrieval metrics
    print("\nComputing retrieval metrics...")

    results = {}
    for threshold in [0.7, 0.8, 0.9]:
        recalls = compute_recall_at_k(embeddings, similarity_matrix, k_values=[1, 5, 10], threshold=threshold)
        map_score = compute_map(embeddings, similarity_matrix, threshold=threshold)

        results[f'threshold_{threshold}'] = {
            'recall_at_1': recalls[1],
            'recall_at_5': recalls[5],
            'recall_at_10': recalls[10],
            'mAP': map_score
        }

    # Print results
    print(f"\n{'='*70}")
    print("RETRIEVAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'mAP':>8}")
    print("-" * 70)

    for thresh_key, metrics in results.items():
        thresh = thresh_key.split('_')[1]
        print(f"{thresh:<12} {metrics['recall_at_1']:>8.4f} {metrics['recall_at_5']:>8.4f} "
              f"{metrics['recall_at_10']:>8.4f} {metrics['mAP']:>8.4f}")

    # Create visualization
    print("\nCreating retrieval visualization...")

    # Select diverse query samples (one per condition)
    condition_samples = defaultdict(list)
    for i, key in enumerate(keys):
        condition_samples[key[1]].append(i)

    query_indices = []
    for condition in ['uncover', 'cover1', 'cover2']:
        if condition in condition_samples and len(condition_samples[condition]) > 0:
            query_indices.append(np.random.choice(condition_samples[condition]))

    if query_indices:
        create_retrieval_visualization(
            embeddings, keys, ir_paths, pm_paths, similarity_matrix,
            query_indices, k=5,
            save_path=RESULT_DIR / "retrieval_visualization.png"
        )

    # Save results
    with open(RESULT_DIR / "retrieval_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save embeddings for later use
    np.savez(RESULT_DIR / "embeddings.npz",
             embeddings=embeddings,
             keys=np.array([f"{k[0]}_{k[1]}_{k[2]}" for k in keys]))

    print(f"\nResults saved to: {RESULT_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

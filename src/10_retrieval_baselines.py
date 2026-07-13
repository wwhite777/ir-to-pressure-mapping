#!/usr/bin/env python3
"""
10_retrieval_baselines.py - Retrieval Baselines for ICMR 2026

Compares our regression-trained encoder features against:
1. Raw IR pixel similarity (L2 distance)
2. Pretrained ResNet50 (ImageNet weights)
3. PCA on raw pixels
4. Predicted pressure map similarity (cosine)
5. Predicted pressure map SSIM
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
import cv2

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


def compute_retrieval_metrics(query_features, gallery_features, relevance_matrix,
                               k_values=[1, 5, 10]):
    """
    Compute retrieval metrics.

    Args:
        query_features: (N_q, D) numpy array
        gallery_features: (N_g, D) numpy array
        relevance_matrix: (N_q, N_g) boolean array where True = relevant pair

    Returns:
        dict with R@k and mAP
    """
    # Compute similarity
    similarity = cosine_similarity(query_features, gallery_features)

    n_queries = len(query_features)
    recalls = {k: 0.0 for k in k_values}
    aps = []

    for i in range(n_queries):
        # Sort gallery by similarity
        sorted_indices = np.argsort(similarity[i])[::-1]

        # Find relevant items based on relevance matrix
        relevant_positions = [j for j, idx in enumerate(sorted_indices) if relevance_matrix[i, idx]]

        if len(relevant_positions) == 0:
            continue

        # Compute R@k
        for k in k_values:
            if any(r < k for r in relevant_positions):
                recalls[k] += 1

        # Compute AP
        precisions = []
        n_rel_found = 0
        for j, idx in enumerate(sorted_indices):
            if relevance_matrix[i, idx]:
                n_rel_found += 1
                precisions.append(n_rel_found / (j + 1))

        if precisions:
            aps.append(np.mean(precisions))

    # Average
    for k in k_values:
        recalls[k] /= n_queries

    mAP = np.mean(aps) if aps else 0.0

    return {f'R@{k}': recalls[k] for k in k_values}, mAP


def compute_pressure_similarity(pm1, pm2, method='ssim'):
    """Compute similarity between two pressure maps."""
    if method == 'ssim':
        return ssim(pm1, pm2, data_range=1.0)
    elif method == 'cosine':
        return np.dot(pm1.flatten(), pm2.flatten()) / (
            np.linalg.norm(pm1.flatten()) * np.linalg.norm(pm2.flatten()) + 1e-8)
    elif method == 'mse':
        return -np.mean((pm1 - pm2) ** 2)  # Negative because higher is more similar
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    print(f"ICMR 2026 - Retrieval Baselines")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")

    # 1. Our trained model (for encoder features)
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    our_model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    our_model.load_state_dict(state_dict, strict=False)
    our_model = our_model.to(device)
    our_model.eval()
    print("  Our model loaded")

    # 2. Pretrained ResNet50 (ImageNet)
    pretrained_resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    pretrained_resnet = nn.Sequential(*list(pretrained_resnet.children())[:-1])  # Remove classifier
    pretrained_resnet = pretrained_resnet.to(device)
    pretrained_resnet.eval()
    print("  Pretrained ResNet50 loaded")

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])
    dataset = IR_PM_Dataset(IR_PATH, PM_PATH, transform)
    print(f"\nDataset: {len(dataset)} samples")

    # Split into query and gallery (subject-independent)
    all_subjects = list(set(k[0] for _, _, k in dataset.pairs))
    np.random.seed(42)
    np.random.shuffle(all_subjects)

    # Use first 20% subjects as query, rest as gallery
    n_query_subjects = max(10, len(all_subjects) // 5)
    query_subjects = set(all_subjects[:n_query_subjects])
    gallery_subjects = set(all_subjects[n_query_subjects:])

    query_indices = [i for i, (_, _, k) in enumerate(dataset.pairs) if k[0] in query_subjects]
    gallery_indices = [i for i, (_, _, k) in enumerate(dataset.pairs) if k[0] in gallery_subjects]

    print(f"Query subjects: {len(query_subjects)}, samples: {len(query_indices)}")
    print(f"Gallery subjects: {len(gallery_subjects)}, samples: {len(gallery_indices)}")

    # Limit for efficiency
    query_indices = query_indices[:200]
    gallery_indices = gallery_indices[:800]

    print(f"Using {len(query_indices)} queries, {len(gallery_indices)} gallery samples")

    # Extract features for all baselines
    print("\nExtracting features...")

    query_raw_pixels = []
    gallery_raw_pixels = []
    query_our_features = []
    gallery_our_features = []
    query_pretrained_features = []
    gallery_pretrained_features = []
    query_pm = []
    gallery_pm = []
    query_pred_pm = []
    gallery_pred_pm = []
    query_labels = []
    gallery_labels = []

    # ImageNet normalization for pretrained
    imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Extract query features
    print("  Extracting query features...")
    for idx in tqdm(query_indices, desc="Query"):
        ir, pm, key = dataset[idx]
        ir = ir.unsqueeze(0).to(device)

        with torch.no_grad():
            # Raw pixels
            query_raw_pixels.append(ir.flatten().cpu().numpy())

            # Our encoder features
            encoder_features = our_model.encoder(ir)
            bottleneck = encoder_features[-1]  # Deepest features
            query_our_features.append(bottleneck.flatten().cpu().numpy())

            # Our prediction
            pred = our_model(ir).squeeze().cpu().numpy()
            pred = cv2.resize(pred, (pm.shape[2], pm.shape[1]))
            query_pred_pm.append(pred)

            # Pretrained ResNet features
            ir_normalized = imagenet_normalize(ir.squeeze(0)).unsqueeze(0)
            pretrained_feat = pretrained_resnet(ir_normalized)
            query_pretrained_features.append(pretrained_feat.flatten().cpu().numpy())

        query_pm.append(pm.squeeze().numpy())
        query_labels.append(key[0])  # Subject ID

    # Extract gallery features
    print("  Extracting gallery features...")
    for idx in tqdm(gallery_indices, desc="Gallery"):
        ir, pm, key = dataset[idx]
        ir = ir.unsqueeze(0).to(device)

        with torch.no_grad():
            # Raw pixels
            gallery_raw_pixels.append(ir.flatten().cpu().numpy())

            # Our encoder features
            encoder_features = our_model.encoder(ir)
            bottleneck = encoder_features[-1]
            gallery_our_features.append(bottleneck.flatten().cpu().numpy())

            # Our prediction
            pred = our_model(ir).squeeze().cpu().numpy()
            pred = cv2.resize(pred, (pm.shape[2], pm.shape[1]))
            gallery_pred_pm.append(pred)

            # Pretrained ResNet features
            ir_normalized = imagenet_normalize(ir.squeeze(0)).unsqueeze(0)
            pretrained_feat = pretrained_resnet(ir_normalized)
            gallery_pretrained_features.append(pretrained_feat.flatten().cpu().numpy())

        gallery_pm.append(pm.squeeze().numpy())
        gallery_labels.append(key[0])

    # Convert to arrays
    query_raw_pixels = np.vstack(query_raw_pixels)
    gallery_raw_pixels = np.vstack(gallery_raw_pixels)
    query_our_features = np.vstack(query_our_features)
    gallery_our_features = np.vstack(gallery_our_features)
    query_pretrained_features = np.vstack(query_pretrained_features)
    gallery_pretrained_features = np.vstack(gallery_pretrained_features)
    query_labels = np.array(query_labels)
    gallery_labels = np.array(gallery_labels)

    # PCA on raw pixels
    print("  Computing PCA...")
    pca = PCA(n_components=256)
    all_pixels = np.vstack([query_raw_pixels, gallery_raw_pixels])
    pca.fit(all_pixels)
    query_pca = pca.transform(query_raw_pixels)
    gallery_pca = pca.transform(gallery_raw_pixels)

    # Compute relevance matrix based on GT pressure map similarity
    print("\nComputing relevance matrix (GT pressure SSIM)...")
    ssim_threshold = 0.7  # Pairs with SSIM > threshold are "relevant"

    relevance_matrix = np.zeros((len(query_pm), len(gallery_pm)), dtype=bool)
    gt_ssim_matrix = np.zeros((len(query_pm), len(gallery_pm)))

    for i, qp in enumerate(tqdm(query_pm, desc="Relevance")):
        for j, gp in enumerate(gallery_pm):
            s = ssim(qp, gp, data_range=1.0)
            gt_ssim_matrix[i, j] = s
            relevance_matrix[i, j] = s > ssim_threshold

    n_relevant_per_query = relevance_matrix.sum(axis=1)
    print(f"  Avg relevant items per query: {n_relevant_per_query.mean():.1f}")
    print(f"  Queries with at least 1 relevant: {(n_relevant_per_query > 0).sum()}/{len(query_pm)}")

    # Compute retrieval metrics for each baseline
    print("\nComputing retrieval metrics...")

    results = {}

    # 1. Raw IR Pixels (L2)
    print("  1. Raw IR Pixels (L2)...")
    from sklearn.metrics.pairwise import euclidean_distances
    l2_dist = euclidean_distances(query_raw_pixels, gallery_raw_pixels)
    similarity = -l2_dist
    recalls, mAP = compute_retrieval_metrics_from_similarity(similarity, relevance_matrix)
    results['Raw IR Pixels (L2)'] = {'recalls': recalls, 'mAP': mAP}
    print(f"     R@1={recalls['R@1']:.4f}, R@5={recalls['R@5']:.4f}, mAP={mAP:.4f}")

    # 2. Pretrained ResNet50
    print("  2. Pretrained ResNet50...")
    recalls, mAP = compute_retrieval_metrics(
        query_pretrained_features, gallery_pretrained_features, relevance_matrix)
    results['Pretrained ResNet50'] = {'recalls': recalls, 'mAP': mAP}
    print(f"     R@1={recalls['R@1']:.4f}, R@5={recalls['R@5']:.4f}, mAP={mAP:.4f}")

    # 3. PCA + Cosine
    print("  3. PCA + Cosine...")
    recalls, mAP = compute_retrieval_metrics(query_pca, gallery_pca, relevance_matrix)
    results['PCA + Cosine'] = {'recalls': recalls, 'mAP': mAP}
    print(f"     R@1={recalls['R@1']:.4f}, R@5={recalls['R@5']:.4f}, mAP={mAP:.4f}")

    # 4. Predicted PM Cosine
    print("  4. Predicted PM Cosine...")
    query_pred_flat = np.vstack([p.flatten() for p in query_pred_pm])
    gallery_pred_flat = np.vstack([p.flatten() for p in gallery_pred_pm])
    recalls, mAP = compute_retrieval_metrics(query_pred_flat, gallery_pred_flat, relevance_matrix)
    results['Predicted PM Cosine'] = {'recalls': recalls, 'mAP': mAP}
    print(f"     R@1={recalls['R@1']:.4f}, R@5={recalls['R@5']:.4f}, mAP={mAP:.4f}")

    # 5. Predicted PM SSIM (use precomputed similarity matrix if we predict on predictions)
    print("  5. Predicted PM SSIM...")
    pred_ssim_similarity = np.zeros((len(query_pred_pm), len(gallery_pred_pm)))
    for i, qp in enumerate(tqdm(query_pred_pm, desc="SSIM")):
        for j, gp in enumerate(gallery_pred_pm):
            pred_ssim_similarity[i, j] = ssim(qp, gp, data_range=1.0)
    recalls, mAP = compute_retrieval_metrics_from_similarity(pred_ssim_similarity, relevance_matrix)
    results['Predicted PM SSIM'] = {'recalls': recalls, 'mAP': mAP}
    print(f"     R@1={recalls['R@1']:.4f}, R@5={recalls['R@5']:.4f}, mAP={mAP:.4f}")

    # 6. Our Encoder Features (Ours)
    print("  6. Our Encoder Features...")
    recalls, mAP = compute_retrieval_metrics(query_our_features, gallery_our_features, relevance_matrix)
    results['Ours (Encoder)'] = {'recalls': recalls, 'mAP': mAP}
    print(f"     R@1={recalls['R@1']:.4f}, R@5={recalls['R@5']:.4f}, mAP={mAP:.4f}")

    # Save results
    print(f"\n{'='*70}")
    print("RETRIEVAL BASELINES COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'mAP':>8}")
    print("-" * 70)

    for method, data in results.items():
        r = data['recalls']
        print(f"{method:<25} {r['R@1']:>8.4f} {r['R@5']:>8.4f} {r['R@10']:>8.4f} {data['mAP']:>8.4f}")

    # Save to JSON
    with open(RESULT_DIR / "baseline_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULT_DIR / 'baseline_comparison.json'}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def compute_retrieval_metrics_from_similarity(similarity, relevance_matrix, k_values=[1, 5, 10]):
    """Compute retrieval metrics from precomputed similarity matrix."""
    n_queries = similarity.shape[0]
    recalls = {k: 0.0 for k in k_values}
    aps = []

    for i in range(n_queries):
        sorted_indices = np.argsort(similarity[i])[::-1]

        # Find positions of relevant items
        relevant_positions = [j for j, idx in enumerate(sorted_indices)
                              if relevance_matrix[i, idx]]

        if len(relevant_positions) == 0:
            continue

        # R@k
        for k in k_values:
            if any(r < k for r in relevant_positions):
                recalls[k] += 1

        # AP
        precisions = []
        n_rel_found = 0
        for j, idx in enumerate(sorted_indices):
            if relevance_matrix[i, idx]:
                n_rel_found += 1
                precisions.append(n_rel_found / (j + 1))

        if precisions:
            aps.append(np.mean(precisions))

    for k in k_values:
        recalls[k] /= n_queries

    mAP = np.mean(aps) if aps else 0.0

    return {f'R@{k}': recalls[k] for k in k_values}, mAP


if __name__ == "__main__":
    main()

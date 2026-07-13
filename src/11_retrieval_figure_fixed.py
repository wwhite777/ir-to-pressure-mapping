#!/usr/bin/env python3
"""
11_retrieval_figure_fixed.py - Fixed Retrieval Figure with Pressure Maps

Creates Figure 4 showing:
- Query: IR image + Pressure map
- Top-5 Retrieved: IR images + Pressure maps
- Similarity scores with relevance indicators
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
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def load_data():
    """Load IR and PM data."""
    pattern = re.compile(
        r"^(?P<subject>\d+)_(?:ir|lr|pm)_(?P<condition>[A-Za-z0-9]+)_image_(?P<frame>\d+)\.png$",
        re.IGNORECASE
    )

    def parse_key(filename):
        m = pattern.match(filename)
        if not m:
            return None
        return (int(m.group('subject')), m.group('condition').lower(), int(m.group('frame')))

    ir_files = {}
    pm_files = {}

    for f in os.listdir(IR_PATH):
        key = parse_key(f)
        if key:
            ir_files[key] = IR_PATH / f

    for f in os.listdir(PM_PATH):
        key = parse_key(f)
        if key:
            pm_files[key] = PM_PATH / f

    common_keys = sorted(set(ir_files.keys()) & set(pm_files.keys()))
    return [(ir_files[k], pm_files[k], k) for k in common_keys]


def load_and_preprocess(ir_path, pm_path):
    """Load and preprocess IR and PM images."""
    # Load IR
    ir_img = Image.open(ir_path).convert('RGB')
    ir_raw = np.array(ir_img)
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_norm = np.power(ir_norm, 0.75)
    ir_display = np.clip(ir_norm * 255, 0, 255).astype(np.uint8)

    # Load PM
    pm_img = Image.open(pm_path)
    pm_raw = np.array(pm_img)
    if len(pm_raw.shape) == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    pm_norm = pm_raw.astype(np.float32) / 255.0

    return ir_display, pm_norm


def main():
    print(f"Creating Fixed Retrieval Figure")
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

    # Load data
    data = load_data()
    print(f"Loaded {len(data)} samples")

    # Split by subject for subject-independent evaluation
    all_subjects = list(set(k[0] for _, _, k in data))
    np.random.seed(42)
    np.random.shuffle(all_subjects)

    query_subjects = set(all_subjects[:15])
    gallery_subjects = set(all_subjects[15:])

    query_data = [(ir, pm, k) for ir, pm, k in data if k[0] in query_subjects]
    gallery_data = [(ir, pm, k) for ir, pm, k in data if k[0] in gallery_subjects]

    # Limit for efficiency
    query_data = query_data[:100]
    gallery_data = gallery_data[:500]

    print(f"Query: {len(query_data)}, Gallery: {len(gallery_data)}")

    # Extract embeddings
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    def extract_features(data_list):
        embeddings = []
        ir_images = []
        pm_images = []

        for ir_path, pm_path, key in data_list:
            ir_display, pm_norm = load_and_preprocess(ir_path, pm_path)
            ir_images.append(ir_display)
            pm_images.append(pm_norm)

            # Get embedding
            ir_tensor = transform(ir_display).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encoder(ir_tensor)
                embedding = features[-1].mean(dim=[2, 3]).flatten().cpu().numpy()
            embeddings.append(embedding)

        return np.vstack(embeddings), ir_images, pm_images

    print("Extracting query features...")
    query_embeddings, query_ir, query_pm = extract_features(query_data)

    print("Extracting gallery features...")
    gallery_embeddings, gallery_ir, gallery_pm = extract_features(gallery_data)

    # Compute similarity
    similarity = cosine_similarity(query_embeddings, gallery_embeddings)

    # Find a good query example (one with mixed relevant/irrelevant in top-5)
    ssim_threshold = 0.7

    best_query_idx = None
    for q_idx in range(len(query_data)):
        top5_indices = np.argsort(similarity[q_idx])[::-1][:5]

        # Check relevance
        relevance = []
        for g_idx in top5_indices:
            s = ssim(query_pm[q_idx], gallery_pm[g_idx], data_range=1.0)
            relevance.append(s > ssim_threshold)

        # Want mix of relevant and irrelevant
        if 2 <= sum(relevance) <= 4:
            best_query_idx = q_idx
            break

    if best_query_idx is None:
        best_query_idx = 0

    # Create figure
    fig = plt.figure(figsize=(16, 6))

    # Query column
    ax_query_ir = fig.add_axes([0.02, 0.55, 0.12, 0.35])
    ax_query_pm = fig.add_axes([0.02, 0.10, 0.12, 0.35])

    # Query IR
    ax_query_ir.imshow(query_ir[best_query_idx])
    ax_query_ir.set_title('Query IR', fontsize=11, fontweight='bold')
    ax_query_ir.axis('off')

    # Query PM
    im_q = ax_query_pm.imshow(query_pm[best_query_idx], cmap='jet', vmin=0, vmax=1)
    ax_query_pm.set_title('Query Pressure', fontsize=11, fontweight='bold')
    ax_query_pm.axis('off')

    # Arrow
    fig.text(0.155, 0.5, '→', fontsize=30, ha='center', va='center')

    # Top-5 retrieved
    top5_indices = np.argsort(similarity[best_query_idx])[::-1][:5]

    for i, g_idx in enumerate(top5_indices):
        x_start = 0.18 + i * 0.16

        # IR image
        ax_ir = fig.add_axes([x_start, 0.55, 0.14, 0.35])
        ax_ir.imshow(gallery_ir[g_idx])
        ax_ir.axis('off')

        # PM image
        ax_pm = fig.add_axes([x_start, 0.10, 0.14, 0.35])
        ax_pm.imshow(gallery_pm[g_idx], cmap='jet', vmin=0, vmax=1)
        ax_pm.axis('off')

        # Compute SSIM for relevance
        pm_ssim = ssim(query_pm[best_query_idx], gallery_pm[g_idx], data_range=1.0)
        cosine_sim = similarity[best_query_idx, g_idx]
        is_relevant = pm_ssim > ssim_threshold

        # Add border color based on relevance
        border_color = 'green' if is_relevant else 'red'
        for ax in [ax_ir, ax_pm]:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
                spine.set_visible(True)

        # Title with similarity
        relevance_symbol = '✓' if is_relevant else '✗'
        ax_ir.set_title(f'#{i+1} Cos:{cosine_sim:.2f}', fontsize=10)
        ax_pm.set_title(f'SSIM:{pm_ssim:.2f} {relevance_symbol}', fontsize=10,
                       color='green' if is_relevant else 'red')

    # Add legend
    fig.text(0.5, 0.02,
             'Green border = Relevant (SSIM > 0.7)  |  Red border = Irrelevant (SSIM ≤ 0.7)  |  Cos = Cosine similarity (encoder features)',
             ha='center', fontsize=10, style='italic')

    plt.suptitle('Cross-Modal Retrieval: Query IR → Retrieved Pressure Patterns',
                fontsize=14, fontweight='bold', y=0.98)

    save_path = RESULT_DIR / "retrieval_with_pressure_maps.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

    print("Done!")


if __name__ == "__main__":
    main()

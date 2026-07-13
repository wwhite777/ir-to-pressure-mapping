#!/usr/bin/env python3
"""
19_final_fixes.py - Fix remaining issues:
1. Create Figure 1 (method overview)
2. Regenerate Figure 4 with a successful retrieval case (not failure)
"""

import os
import numpy as np
from pathlib import Path
import json
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from skimage.metrics import structural_similarity as ssim
import cv2
import re

PROJECT_ROOT = Path("/home/wjeong/cc")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "figure" / "final"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IR_PATH = DATA_ROOT / "IR_png"
PM_PATH = DATA_ROOT / "PM_png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_figure1_method_overview():
    """Create Figure 1: Method overview diagram."""
    print("Creating Figure 1: Method Overview...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors
    input_color = '#3498db'
    encoder_color = '#2ecc71'
    decoder_color = '#e74c3c'
    uncertainty_color = '#9b59b6'
    output_color = '#f39c12'
    retrieval_color = '#1abc9c'

    # Input IR Image
    rect = FancyBboxPatch((0.3, 2), 1.8, 2, boxstyle="round,pad=0.1",
                          facecolor=input_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(1.2, 3, 'IR\nImage', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Arrow to encoder
    ax.annotate('', xy=(2.5, 3), xytext=(2.1, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # U-Net Encoder
    rect = FancyBboxPatch((2.5, 1.5), 2.5, 3, boxstyle="round,pad=0.1",
                          facecolor=encoder_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(3.75, 3, 'U-Net\nEncoder\n(ResNet-50)', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Branch 1: To Decoder
    ax.annotate('', xy=(5.5, 3.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Branch 2: To Embedding (for retrieval)
    ax.plot([5, 5.3, 5.3], [2.5, 2.5, 1], 'k-', lw=2)
    ax.annotate('', xy=(5.8, 1), xytext=(5.3, 1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Embedding for retrieval
    rect = FancyBboxPatch((5.8, 0.3), 2, 1.4, boxstyle="round,pad=0.1",
                          facecolor=retrieval_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(6.8, 1, 'Embedding\n(2048-d)', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Arrow to retrieval
    ax.annotate('', xy=(8.3, 1), xytext=(7.8, 1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Retrieval output
    rect = FancyBboxPatch((8.3, 0.3), 2.2, 1.4, boxstyle="round,pad=0.1",
                          facecolor=retrieval_color, edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(rect)
    ax.text(9.4, 1, 'Cross-Modal\nRetrieval', ha='center', va='center', fontsize=9, fontweight='bold')

    # U-Net Decoder with MC Dropout
    rect = FancyBboxPatch((5.5, 2.5), 2.5, 2, boxstyle="round,pad=0.1",
                          facecolor=decoder_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(6.75, 3.5, 'U-Net Decoder\n+ MC Dropout', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Arrow to MC samples
    ax.annotate('', xy=(8.5, 3.5), xytext=(8, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # MC Dropout samples (N=20)
    rect = FancyBboxPatch((8.5, 2.5), 2, 2, boxstyle="round,pad=0.1",
                          facecolor=uncertainty_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(9.5, 3.5, 'N=20\nSamples', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Two outputs from MC samples
    # Mean prediction
    ax.plot([10.5, 11, 11], [4, 4, 4.5], 'k-', lw=2)
    ax.annotate('', xy=(11.5, 4.5), xytext=(11, 4.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Uncertainty
    ax.plot([10.5, 11, 11], [3, 3, 2], 'k-', lw=2)
    ax.annotate('', xy=(11.5, 2), xytext=(11, 2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Mean prediction box
    rect = FancyBboxPatch((11.5, 4), 2, 1.2, boxstyle="round,pad=0.1",
                          facecolor=output_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(12.5, 4.6, 'Mean\nPrediction', ha='center', va='center', fontsize=9, fontweight='bold')

    # Uncertainty box
    rect = FancyBboxPatch((11.5, 1.4), 2, 1.2, boxstyle="round,pad=0.1",
                          facecolor=uncertainty_color, edgecolor='black', linewidth=2, alpha=0.6)
    ax.add_patch(rect)
    ax.text(12.5, 2, 'Uncertainty\n(Std)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Adaptive EMA
    ax.plot([12.5, 12.5], [4, 3.2], 'k-', lw=2)
    ax.plot([12.5, 12.5], [2.6, 3.2], 'k-', lw=2)

    # Final output arrow
    ax.annotate('', xy=(12.5, 5.5), xytext=(12.5, 5.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Adaptive EMA label
    ax.text(12.5, 3.4, 'Adaptive\nEMA', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # Final output
    rect = FancyBboxPatch((11.3, 5.5), 2.4, 0.4, boxstyle="round,pad=0.05",
                          facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(12.5, 5.7, 'Stabilized Pressure Map', ha='center', va='center', fontsize=9, fontweight='bold')

    # Title
    ax.text(7, 5.8, 'Figure 1: Method Overview', ha='center', va='center', fontsize=14, fontweight='bold')

    # Legend
    legend_y = 0.3
    legend_items = [
        (input_color, 'Input'),
        (encoder_color, 'Encoder'),
        (decoder_color, 'Decoder'),
        (uncertainty_color, 'Uncertainty'),
        (output_color, 'Output'),
        (retrieval_color, 'Retrieval')
    ]
    for i, (color, label) in enumerate(legend_items):
        x = 0.3 + i * 1.8
        rect = FancyBboxPatch((x, legend_y - 0.15), 0.3, 0.3, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y, label, fontsize=8, va='center')

    save_path = RESULT_DIR / "figure1_method_overview.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


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
    ir_img = Image.open(ir_path).convert('RGB')
    ir_raw = np.array(ir_img)
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_norm = np.power(ir_norm, 0.75)
    ir_display = np.clip(ir_norm * 255, 0, 255).astype(np.uint8)

    pm_img = Image.open(pm_path)
    pm_raw = np.array(pm_img)
    if len(pm_raw.shape) == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    pm_norm = pm_raw.astype(np.float32) / 255.0

    return ir_display, pm_norm


def create_figure4_retrieval_success(model, data):
    """Create Figure 4 with a SUCCESSFUL retrieval case."""
    print("Creating Figure 4: Retrieval (Success Case)...")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    # Split data
    all_subjects = list(set(k[0] for _, _, k in data))
    np.random.seed(42)
    np.random.shuffle(all_subjects)

    query_subjects = set(all_subjects[:15])
    gallery_subjects = set(all_subjects[15:])

    query_data = [(ir, pm, k) for ir, pm, k in data if k[0] in query_subjects][:100]
    gallery_data = [(ir, pm, k) for ir, pm, k in data if k[0] in gallery_subjects][:500]

    # Extract features
    def extract_features(data_list):
        embeddings = []
        ir_images = []
        pm_images = []

        for ir_path, pm_path, key in data_list:
            ir_display, pm_norm = load_and_preprocess(ir_path, pm_path)
            ir_images.append(ir_display)
            pm_images.append(pm_norm)

            ir_tensor = transform(ir_display).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encoder(ir_tensor)
                embedding = features[-1].mean(dim=[2, 3]).flatten().cpu().numpy()
            embeddings.append(embedding)

        return np.vstack(embeddings), ir_images, pm_images

    query_embeddings, query_ir, query_pm = extract_features(query_data)
    gallery_embeddings, gallery_ir, gallery_pm = extract_features(gallery_data)

    # Compute similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(query_embeddings, gallery_embeddings)

    # Find a SUCCESSFUL query (all top-5 are relevant)
    ssim_threshold = 0.7
    best_query_idx = None

    for q_idx in range(len(query_data)):
        top5_indices = np.argsort(similarity[q_idx])[::-1][:5]

        relevance = []
        for g_idx in top5_indices:
            s = ssim(query_pm[q_idx], gallery_pm[g_idx], data_range=1.0)
            relevance.append(s > ssim_threshold)

        # Want mostly relevant (4-5 out of 5)
        if sum(relevance) >= 4:
            best_query_idx = q_idx
            print(f"Found successful query at index {q_idx} with {sum(relevance)}/5 relevant")
            break

    if best_query_idx is None:
        # Fallback: find best available
        best_score = 0
        for q_idx in range(len(query_data)):
            top5_indices = np.argsort(similarity[q_idx])[::-1][:5]
            relevance = [ssim(query_pm[q_idx], gallery_pm[g_idx], data_range=1.0) > ssim_threshold
                        for g_idx in top5_indices]
            if sum(relevance) > best_score:
                best_score = sum(relevance)
                best_query_idx = q_idx
        print(f"Using best available query at index {best_query_idx} with {best_score}/5 relevant")

    # Create figure
    fig = plt.figure(figsize=(16, 6))

    # Query column
    ax_query_ir = fig.add_axes([0.02, 0.55, 0.12, 0.35])
    ax_query_pm = fig.add_axes([0.02, 0.10, 0.12, 0.35])

    ax_query_ir.imshow(query_ir[best_query_idx])
    ax_query_ir.set_title('Query IR', fontsize=11, fontweight='bold')
    ax_query_ir.axis('off')

    ax_query_pm.imshow(query_pm[best_query_idx], cmap='jet', vmin=0, vmax=1)
    ax_query_pm.set_title('Query Pressure*', fontsize=11, fontweight='bold')
    ax_query_pm.axis('off')

    fig.text(0.155, 0.5, '→', fontsize=30, ha='center', va='center')

    # Top-5 retrieved
    top5_indices = np.argsort(similarity[best_query_idx])[::-1][:5]

    for i, g_idx in enumerate(top5_indices):
        x_start = 0.18 + i * 0.16

        ax_ir = fig.add_axes([x_start, 0.55, 0.14, 0.35])
        ax_ir.imshow(gallery_ir[g_idx])
        ax_ir.axis('off')

        ax_pm = fig.add_axes([x_start, 0.10, 0.14, 0.35])
        ax_pm.imshow(gallery_pm[g_idx], cmap='jet', vmin=0, vmax=1)
        ax_pm.axis('off')

        pm_ssim = ssim(query_pm[best_query_idx], gallery_pm[g_idx], data_range=1.0)
        cosine_sim = similarity[best_query_idx, g_idx]
        is_relevant = pm_ssim > ssim_threshold

        border_color = 'green' if is_relevant else 'red'
        for ax in [ax_ir, ax_pm]:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
                spine.set_visible(True)

        relevance_symbol = '✓' if is_relevant else '✗'
        ax_ir.set_title(f'#{i+1} Cos:{cosine_sim:.2f}', fontsize=10)
        ax_pm.set_title(f'SSIM:{pm_ssim:.2f} {relevance_symbol}', fontsize=10,
                       color='green' if is_relevant else 'red')

    fig.text(0.5, 0.02,
             'Green = Relevant (SSIM > 0.7) | Red = Irrelevant | *Query pressure shown for evaluation only',
             ha='center', fontsize=10, style='italic')

    plt.suptitle('Cross-Modal Retrieval: Query IR → Retrieved Pressure Patterns',
                fontsize=14, fontweight='bold', y=0.98)

    save_path = RESULT_DIR / "figure4_retrieval.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Creating Final Fixes")
    print("=" * 60)

    # Create Figure 1
    create_figure1_method_overview()

    # Load model for Figure 4
    model_path = MODEL_DIR / "U-Net(ResNet50)_E50.pth"
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    data = load_data()
    print(f"Loaded {len(data)} samples")

    # Create Figure 4 with success case
    create_figure4_retrieval_success(model, data)

    print("\nDone!")


if __name__ == "__main__":
    main()

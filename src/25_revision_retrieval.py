#!/usr/bin/env python3
"""
25_revision_retrieval.py — IEEE Access revision experiments (cross-modal retrieval).

Addresses (Access-2026-33087):
  R1#3 / R2#5  Relevance-threshold sensitivity: R@K and mAP at SSIM thresholds
               {0.60, 0.65, 0.70, 0.75, 0.80}, with relevant-items-per-query
               statistics so the 0.70 choice can be justified.
  R1#3         Scaled-up protocol: 200 queries, full gallery-subject pool
               (~7,900 items) instead of the original 200/800.
  All six methods from the original comparison are re-run identically.

Protocol matches 10_retrieval_baselines.py (seed 42, subject-disjoint
query/gallery, same preprocessing, same feature definitions).

Output: result/revision/retrieval_revision.json
"""

import os
import json
import time
import hashlib
import subprocess
import re
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from skimage.metrics import structural_similarity as ssim
import cv2

PROJECT_ROOT = Path(os.environ.get("CC_PROJECT_ROOT",
                                   Path(__file__).resolve().parent.parent))
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "revision"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_QUERY = 200
GALLERY_CAP = 8000          # effectively the full gallery-subject pool
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
K_VALUES = [1, 5, 10]
N_WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pairs():
    pattern = re.compile(
        r"^(?P<subject>\d+)_(?:ir|lr|pm)_(?P<condition>[A-Za-z0-9]+)_image_(?P<frame>\d+)\.png$",
        re.IGNORECASE)
    ir_map, pm_map = {}, {}
    for f in os.listdir(DATA_ROOT / "IR_png"):
        m = pattern.match(f)
        if m:
            ir_map[(int(m.group('subject')), m.group('condition').lower(),
                    int(m.group('frame')))] = str(DATA_ROOT / "IR_png" / f)
    for f in os.listdir(DATA_ROOT / "PM_png"):
        m = pattern.match(f)
        if m:
            pm_map[(int(m.group('subject')), m.group('condition').lower(),
                    int(m.group('frame')))] = str(DATA_ROOT / "PM_png" / f)
    common = sorted(set(ir_map) & set(pm_map))
    return [(ir_map[k], pm_map[k], k) for k in common]


def load_sample(ir_path, pm_path, transform):
    ir_img = Image.open(ir_path).convert('RGB')
    ir_raw = np.array(ir_img)
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_norm = np.power(ir_norm, 0.75)
    ir_norm = np.clip(ir_norm * 255, 0, 255).astype(np.uint8)
    pm_img = Image.open(pm_path)
    pm_raw = np.array(pm_img)
    if pm_raw.ndim == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    return transform(ir_norm), pm_raw.astype(np.float32) / 255.0


_GALLERY_PM = None


def _init_pool(gallery_pm):
    global _GALLERY_PM
    _GALLERY_PM = gallery_pm


def _ssim_row(args):
    qp, = args
    return np.array([ssim(qp, gp, data_range=1.0) for gp in _GALLERY_PM],
                    dtype=np.float32)


def ssim_matrix(query_maps, gallery_maps, desc):
    with Pool(N_WORKERS, initializer=_init_pool, initargs=(gallery_maps,)) as pool:
        rows = list(tqdm(pool.imap(_ssim_row, [(q,) for q in query_maps]),
                         total=len(query_maps), desc=desc))
    return np.stack(rows)


def metrics_from_similarity(similarity, relevance, k_values=K_VALUES):
    """Same definitions as 10_retrieval_baselines.py: R@k denominator counts all
    queries; AP averaged over queries with >=1 relevant item."""
    n_q = similarity.shape[0]
    recalls = {k: 0.0 for k in k_values}
    aps = []
    for i in range(n_q):
        order = np.argsort(similarity[i])[::-1]
        rel_pos = np.nonzero(relevance[i][order])[0]
        if len(rel_pos) == 0:
            continue
        for k in k_values:
            if rel_pos[0] < k:
                recalls[k] += 1
        hits = np.arange(1, len(rel_pos) + 1)
        aps.append(float(np.mean(hits / (rel_pos + 1))))
    out = {f'R@{k}': recalls[k] / n_q for k in k_values}
    out['mAP'] = float(np.mean(aps)) if aps else 0.0
    return out


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    den = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return [float(center - half), float(center + half)]


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    commit = subprocess.run(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()

    # ---- models ----
    our_model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                         in_channels=3, classes=1)
    ckpt = torch.load(MODEL_DIR / "U-Net(ResNet50)_E50.pth", map_location=device)
    state_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    our_model.load_state_dict(state_dict, strict=False)
    our_model = our_model.to(device).eval()

    pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    pretrained = nn.Sequential(*list(pretrained.children())[:-1]).to(device).eval()
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # ---- split (identical to 10_retrieval_baselines.py) ----
    pairs = load_pairs()
    all_subjects = list(set(k[0] for _, _, k in pairs))
    np.random.seed(SEED)
    np.random.shuffle(all_subjects)
    n_query_subjects = max(10, len(all_subjects) // 5)
    query_subjects = set(all_subjects[:n_query_subjects])
    gallery_subjects = set(all_subjects[n_query_subjects:])

    query_idx = [i for i, (_, _, k) in enumerate(pairs) if k[0] in query_subjects][:N_QUERY]
    gallery_idx = [i for i, (_, _, k) in enumerate(pairs) if k[0] in gallery_subjects][:GALLERY_CAP]
    print(f"Queries: {len(query_idx)}  Gallery: {len(gallery_idx)}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    # ---- feature extraction ----
    def extract(indices, desc):
        feats = {'raw': [], 'ours': [], 'imagenet': [], 'pred_pm': [], 'gt_pm': []}
        for idx in tqdm(indices, desc=desc):
            ir_path, pm_path, key = pairs[idx]
            ir, gt = load_sample(ir_path, pm_path, transform)
            ir = ir.unsqueeze(0).to(device)
            with torch.no_grad():
                feats['raw'].append(ir.flatten().cpu().numpy())
                enc = our_model.encoder(ir)
                feats['ours'].append(enc[-1].flatten().cpu().numpy())
                pred = our_model(ir).squeeze().cpu().numpy()
                feats['pred_pm'].append(cv2.resize(pred, (gt.shape[1], gt.shape[0])))
                feats['imagenet'].append(
                    pretrained(imagenet_norm(ir.squeeze(0)).unsqueeze(0))
                    .flatten().cpu().numpy())
            feats['gt_pm'].append(gt)
        return feats

    qf = extract(query_idx, "query")
    gf = extract(gallery_idx, "gallery")

    q_raw, g_raw = np.vstack(qf['raw']), np.vstack(gf['raw'])
    q_ours, g_ours = np.vstack(qf['ours']), np.vstack(gf['ours'])
    q_im, g_im = np.vstack(qf['imagenet']), np.vstack(gf['imagenet'])

    print("PCA ...")
    pca = PCA(n_components=256, svd_solver='randomized', random_state=SEED)
    pca.fit(np.vstack([q_raw, g_raw]))
    q_pca, g_pca = pca.transform(q_raw), pca.transform(g_raw)

    # ---- similarity matrices ----
    print("GT-PM SSIM relevance matrix ...")
    gt_ssim = ssim_matrix(qf['gt_pm'], gf['gt_pm'], "gt-ssim")
    print("Predicted-PM SSIM similarity ...")
    pred_ssim = ssim_matrix(qf['pred_pm'], gf['pred_pm'], "pred-ssim")

    q_pred_flat = np.vstack([p.flatten() for p in qf['pred_pm']])
    g_pred_flat = np.vstack([p.flatten() for p in gf['pred_pm']])

    sims = {
        'Pretrained ResNet50': cosine_similarity(q_im, g_im),
        'Raw IR Pixels (L2)': -euclidean_distances(q_raw, g_raw),
        'PCA + Cosine': cosine_similarity(q_pca, g_pca),
        'Encoder Features (Ours)': cosine_similarity(q_ours, g_ours),
        'Predicted PM Cosine': cosine_similarity(q_pred_flat, g_pred_flat),
        'Predicted PM SSIM': pred_ssim,
    }

    # ---- encoder-only latency ----
    ir0 = transform(np.zeros((160, 120, 3), dtype=np.uint8)).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(5):
            our_model.encoder(ir0)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            our_model.encoder(ir0)
        torch.cuda.synchronize()
        enc_ms = (time.time() - t) / 50 * 1000

    # ---- evaluate across thresholds ----
    by_threshold = {}
    for th in THRESHOLDS:
        rel = gt_ssim > th
        n_rel = rel.sum(axis=1)
        entry = {
            'avg_relevant_per_query': float(n_rel.mean()),
            'median_relevant_per_query': float(np.median(n_rel)),
            'queries_with_relevant': int((n_rel > 0).sum()),
            'n_queries': int(rel.shape[0]),
            'methods': {},
        }
        for name, sim in sims.items():
            m = metrics_from_similarity(sim, rel)
            m['R@1_ci95'] = wilson_ci(m['R@1'], rel.shape[0])
            entry['methods'][name] = m
        by_threshold[f'{th:.2f}'] = entry
        enc_r1 = entry['methods']['Encoder Features (Ours)']['R@1']
        print(f"th={th:.2f}  avg_rel={entry['avg_relevant_per_query']:.1f}  "
              f"encoder R@1={enc_r1:.3f}")

    results = {
        'meta': {
            'script': '25_revision_retrieval.py', 'git_commit': commit,
            'seed': SEED, 'timestamp': datetime.now().isoformat(),
            'device': torch.cuda.get_device_name(0),
            'n_query': len(query_idx), 'n_gallery': len(gallery_idx),
            'n_query_subjects': len(query_subjects),
            'n_gallery_subjects': len(gallery_subjects),
            'thresholds': THRESHOLDS,
            'config_hash': hashlib.sha256(
                json.dumps([SEED, N_QUERY, GALLERY_CAP, THRESHOLDS]).encode()
            ).hexdigest()[:16],
        },
        'encoder_only_latency_ms': float(enc_ms),
        'by_threshold': by_threshold,
        'runtime_min': (time.time() - t0) / 60,
    }
    out = RESULT_DIR / "retrieval_revision.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}  ({results['runtime_min']:.1f} min)")


if __name__ == "__main__":
    main()

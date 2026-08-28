#!/usr/bin/env python3
"""
26_revision_occlusion_analysis.py — per-condition SSIM distributions + failure analysis.

Addresses (Access-2026-33087):
  R1#1  Occlusion robustness reframed as an empirical finding: report the full
        per-condition SSIM distribution (quantiles, failure fractions), not just
        means, so the thick-blanket failure mode is characterized honestly.

Protocol: identical to Table 1/3 of 16_unified_experiments.py (seed-42 subject
split, base model, 300 samples/condition), extended with per-sample quantiles.

Output: result/revision/occlusion_revision.json
"""

import os
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2

PROJECT_ROOT = Path(os.environ.get("CC_PROJECT_ROOT",
                                   Path(__file__).resolve().parent.parent))
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "revision"
SEED = 42
FAIL_THRESH = 0.65
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    pattern = re.compile(
        r"^(?P<subject>\d+)_(?:ir|lr|pm)_(?P<condition>[A-Za-z0-9]+)_image_(?P<frame>\d+)\.png$",
        re.IGNORECASE)
    ir_map, pm_map = {}, {}
    for f in os.listdir(DATA_ROOT / "IR_png"):
        m = pattern.match(f)
        if m:
            ir_map[(int(m.group('subject')), m.group('condition').lower(),
                    int(m.group('frame')))] = DATA_ROOT / "IR_png" / f
    for f in os.listdir(DATA_ROOT / "PM_png"):
        m = pattern.match(f)
        if m:
            pm_map[(int(m.group('subject')), m.group('condition').lower(),
                    int(m.group('frame')))] = DATA_ROOT / "PM_png" / f
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


def main():
    commit = subprocess.run(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
    ckpt = torch.load(MODEL_DIR / "U-Net(ResNet50)_E50.pth", map_location=device)
    state_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    data = load_data()
    all_subjects = list(set(k[0] for _, _, k in data))
    np.random.seed(SEED)
    np.random.shuffle(all_subjects)
    test_subjects = set(all_subjects[:len(all_subjects) // 5])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])
    by_cond = {'uncover': [], 'cover1': [], 'cover2': []}
    for ir, pm, k in data:
        if k[0] in test_subjects and k[1] in by_cond:
            by_cond[k[1]].append((ir, pm))

    out = {}
    for cond, samples in by_cond.items():
        vals = []
        for ir_path, pm_path in tqdm(samples[:300], desc=cond):
            ir_t, gt = load_sample(ir_path, pm_path, transform)
            with torch.no_grad():
                pred = model(ir_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            pred = np.clip(cv2.resize(pred, (gt.shape[1], gt.shape[0])), 0, 1)
            vals.append(float(ssim(pred, gt, data_range=1.0)))
        v = np.array(vals)
        out[cond] = {
            'n': len(v), 'mean': float(v.mean()), 'std': float(v.std()),
            'p5': float(np.percentile(v, 5)), 'p25': float(np.percentile(v, 25)),
            'median': float(np.median(v)), 'p75': float(np.percentile(v, 75)),
            'p95': float(np.percentile(v, 95)),
            'frac_below_065': float((v < FAIL_THRESH).mean()),
            'frac_below_060': float((v < 0.60).mean()),
            'values': [round(float(x), 4) for x in v],
        }
        print(f"{cond}: mean={v.mean():.4f} p5={np.percentile(v, 5):.3f} "
              f"frac<{FAIL_THRESH}={float((v < FAIL_THRESH).mean()):.3f}")

    results = {'meta': {'script': '26_revision_occlusion_analysis.py',
                        'git_commit': commit, 'seed': SEED,
                        'timestamp': datetime.now().isoformat(),
                        'fail_threshold': FAIL_THRESH},
               'per_condition': out}
    with open(RESULT_DIR / "occlusion_revision.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved occlusion_revision.json")


if __name__ == "__main__":
    main()

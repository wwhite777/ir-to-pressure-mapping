#!/usr/bin/env python3
"""
24_revision_temporal_calibration.py — IEEE Access revision experiments (temporal + calibration).

Addresses (Access-2026-33087):
  R2#1  Kalman honest framing: exhaustive Q/R grid search on a validation split
        (random-walk AND constant-velocity models), evaluated on the SAME unified
        test protocol as every other filter.
  R2#4  MC-dropout uncertainty calibration: reliability curve + ENCE + frame-level
        Spearman correlation between uncertainty and error.
  ICMR-R1 (folded in): N=5 / N=10 / N=20 MC-pass sweep + measured latency.
  R1#5  Dataset statistics per occlusion condition and split.
  Also: statistical significance (paired Wilcoxon, Holm-corrected).

Protocol notes (fixes two spec<->code drifts in the original pipeline):
  * ALL temporal filters (including Kalman) run on the seed-42 subject-independent
    test split of 16_unified_experiments.py. The original Kalman number came from
    15_temporal_baselines.py, which used the first 30 sequences of the WHOLE
    dataset (no split) with N=10 passes and untuned Q=0.01/R=0.1.
  * UA-EMA normalization bounds are the 5th-95th percentile of frame-level
    uncertainty on the VALIDATION split (causal, as the manuscript states),
    not the per-sequence min-max of the test sequence itself (non-causal).

Outputs: result/revision/temporal_revision.json (+ reliability_data.json)
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy import stats as sstats
import cv2

PROJECT_ROOT = Path("/home/wjeong/cc")
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "result" / "revision"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SEQ = 30          # sequences per split (matches 16_unified_experiments.py)
SEQ_LEN = 40        # frames per sequence (matches 16)
MC_MAX = 20
ALPHA_MIN, ALPHA_MAX = 0.1, 0.5
FIXED_ALPHA = 0.3
MA_K = 5
KF_RW_GRID_Q = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
KF_RW_GRID_R = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
KF_CV_GRID_Q = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
KF_CV_GRID_R = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCDropoutWrapper(nn.Module):
    """Same wrapper as 16_unified_experiments.py: Dropout2d(p) on the two deepest
    encoder feature maps and on the decoder output."""

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
        return self.model.segmentation_head(decoder_output)


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def load_data():
    """Identical pairing logic to 16_unified_experiments.py."""
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
    """Identical preprocessing to 16_unified_experiments.py."""
    ir_img = Image.open(ir_path).convert('RGB')
    ir_raw = np.array(ir_img)
    ir_norm = (ir_raw - ir_raw.min()) / (ir_raw.max() - ir_raw.min() + 1e-8)
    ir_norm = np.power(ir_norm, 0.75)
    ir_norm = np.clip(ir_norm * 255, 0, 255).astype(np.uint8)
    pm_img = Image.open(pm_path)
    pm_raw = np.array(pm_img)
    if pm_raw.ndim == 3:
        pm_raw = cv2.cvtColor(pm_raw, cv2.COLOR_RGB2GRAY)
    pm_norm = pm_raw.astype(np.float32) / 255.0
    return transform(ir_norm), pm_norm


def compute_jitter(preds):
    if len(preds) < 2:
        return 0.0
    return float(np.mean([np.abs(preds[i] - preds[i - 1]).mean()
                          for i in range(1, len(preds))]))


def seq_ssim(preds, gts):
    return float(np.mean([ssim(np.clip(p, 0, 1), g, data_range=1.0)
                          for p, g in zip(preds, gts)]))


# ---------------------------------------------------------------- filters ----
def filt_moving_average(preds, k=MA_K):
    return [np.mean(preds[max(0, i - k + 1):i + 1], axis=0) for i in range(len(preds))]


def filt_ema(preds, alphas):
    """EMA with per-frame alpha (scalar list) or fixed alpha (float)."""
    if np.isscalar(alphas):
        alphas = [alphas] * len(preds)
    out, state = [], None
    for p, a in zip(preds, alphas):
        state = p.copy() if state is None else a * p + (1 - a) * state
        out.append(state.copy())
    return out


def ua_ema_alphas(uncertainties, u_lo, u_hi):
    """Causal UA-EMA: bounds are fixed percentiles from the validation split."""
    alphas = []
    for u in uncertainties:
        u_norm = np.clip((u - u_lo) / (u_hi - u_lo + 1e-8), 0.0, 1.0)
        alphas.append(ALPHA_MAX - u_norm * (ALPHA_MAX - ALPHA_MIN))
    return alphas


def filt_kalman_rw(preds, Q, R):
    """Random-walk Kalman filter. The covariance recursion is data-independent,
    so P and the gain K are scalars shared by all pixels (as in the original
    15_temporal_baselines.py implementation)."""
    out, x, P = [], None, 1.0
    for z in preds:
        if x is None:
            x = z.copy()
        else:
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x + K * (z - x)
            P = (1 - K) * P_pred
        out.append(x.copy())
    return out


def filt_kalman_cv(preds, q, R, dt=1.0):
    """Constant-velocity Kalman filter, per-pixel state [x, v] with shared 2x2
    covariance (data-independent recursion). White-noise-acceleration Q."""
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Qm = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    H = np.array([[1.0, 0.0]])
    out = []
    x = None
    v = None
    P = np.eye(2)
    for z in preds:
        if x is None:
            x, v = z.copy(), np.zeros_like(z)
        else:
            # predict
            x_pred = x + dt * v
            v_pred = v
            P = F @ P @ F.T + Qm
            # update (S and K are scalars/2-vectors shared across pixels)
            S = P[0, 0] + R
            K0, K1 = P[0, 0] / S, P[1, 0] / S
            innov = z - x_pred
            x = x_pred + K0 * innov
            v = v_pred + K1 * innov
            P = (np.eye(2) - np.array([[K0], [K1]]) @ H) @ P
        out.append(x.copy())
    return out


# ------------------------------------------------------------ inference ------
def mc_forward(mc_model, ir_tensor, n):
    """One batched forward with n replicas = n independent dropout masks."""
    mc_model.eval()
    enable_dropout(mc_model)
    with torch.no_grad():
        batch = ir_tensor.repeat(n, 1, 1, 1)
        preds = mc_model(batch).squeeze(1).cpu().numpy()
    return preds  # (n, H, W) at model resolution


def collect_split(seq_keys, test_seqs, mc_model, transform, desc):
    """Run MC inference over sequences; return per-sequence dicts with
    mean preds (at PM resolution), sigma maps, scalar uncertainties, GTs,
    plus N=5/N=10 variants (first-k subsets of the same 20 passes)."""
    out = []
    for seq_key in tqdm(seq_keys, desc=desc):
        frames = test_seqs[seq_key][:SEQ_LEN]
        rec = {'key': seq_key, 'mean20': [], 'sig20': [], 'u20': [],
               'mean10': [], 'u10': [], 'mean5': [], 'u5': [], 'gt': []}
        for _, ir_path, pm_path in frames:
            ir_tensor, pm_gt = load_sample(ir_path, pm_path, transform)
            ir_tensor = ir_tensor.unsqueeze(0).to(device)
            passes = mc_forward(mc_model, ir_tensor, MC_MAX)  # (20, 192, 96)
            gt_h, gt_w = pm_gt.shape
            for n_sub, mk, uk in ((20, 'mean20', 'u20'), (10, 'mean10', 'u10'),
                                  (5, 'mean5', 'u5')):
                sub = passes[:n_sub]
                mean_p = cv2.resize(sub.mean(axis=0), (gt_w, gt_h))
                rec[mk].append(mean_p)
                rec[uk].append(float(sub.std(axis=0).mean()))
                if n_sub == 20:
                    rec['sig20'].append(cv2.resize(sub.std(axis=0), (gt_w, gt_h)))
            rec['gt'].append(pm_gt)
        out.append(rec)
    return out


def evaluate_filters(split_recs, u_bounds, kf_rw, kf_cv, mean_key='mean20', u_key='u20'):
    """Apply every filter to every sequence; return per-sequence metric lists."""
    methods = {}

    def add(name, ssim_v, jit_v):
        methods.setdefault(name, {'ssim': [], 'jitter': []})
        methods[name]['ssim'].append(ssim_v)
        methods[name]['jitter'].append(jit_v)

    for rec in split_recs:
        preds, gts, uncs = rec[mean_key], rec['gt'], rec[u_key]
        variants = {
            'no_filter': preds,
            'moving_avg': filt_moving_average(preds),
            'fixed_ema': filt_ema(preds, FIXED_ALPHA),
            'kalman_rw': filt_kalman_rw(preds, *kf_rw),
            'kalman_cv': filt_kalman_cv(preds, *kf_cv),
            'ua_ema': filt_ema(preds, ua_ema_alphas(uncs, *u_bounds)),
        }
        for name, fp in variants.items():
            add(name, seq_ssim(fp, gts), compute_jitter(fp))
    return methods


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    commit = subprocess.run(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    config = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, (int, float, list, str))}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str)
                                 .encode()).hexdigest()[:16]

    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    # ---- model ----
    base_model = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                          in_channels=3, classes=1)
    ckpt = torch.load(MODEL_DIR / "U-Net(ResNet50)_E50.pth", map_location=device)
    state_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                  if not k.endswith(('total_ops', 'total_params'))}
    base_model.load_state_dict(state_dict, strict=False)
    base_model = base_model.to(device).eval()

    mc_model = MCDropoutWrapper(
        smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1), p=0.1)
    mc_model.model.load_state_dict(state_dict, strict=False)
    mc_model = mc_model.to(device)

    # ---- data + split (identical to 16_unified_experiments.py) ----
    data = load_data()
    all_subjects = list(set(k[0] for _, _, k in data))
    np.random.seed(SEED)
    np.random.shuffle(all_subjects)
    n_test = len(all_subjects) // 5
    test_subjects = set(all_subjects[:n_test])
    val_subjects = set(all_subjects[n_test:2 * n_test])  # tuning split (train-side)

    cond_counts = {}
    for _, _, k in data:
        cond_counts[k[1]] = cond_counts.get(k[1], 0) + 1
    dataset_stats = {
        'total_pairs': len(data),
        'n_subjects': len(all_subjects),
        'per_condition': cond_counts,
        'n_test_subjects': len(test_subjects),
        'n_val_subjects': len(val_subjects),
        'test_subjects': sorted(test_subjects),
        'per_condition_test': {},
        'ir_native_wh': [120, 160], 'pm_native_wh': [84, 192],
        'model_input_hw': [192, 96],
    }
    for _, _, k in data:
        if k[0] in test_subjects:
            dataset_stats['per_condition_test'][k[1]] = \
                dataset_stats['per_condition_test'].get(k[1], 0) + 1

    def build_seqs(subject_set):
        seqs = {}
        for ir, pm, k in data:
            if k[0] in subject_set:
                seqs.setdefault((k[0], k[1]), []).append((k[2], ir, pm))
        for sk in seqs:
            seqs[sk].sort(key=lambda x: x[0])
        keys = [k for k in seqs if len(seqs[k]) >= 20][:N_SEQ]
        return seqs, keys

    test_seqs, test_keys = build_seqs(test_subjects)
    val_seqs, val_keys = build_seqs(val_subjects)
    print(f"Test sequences: {len(test_keys)}  Val sequences: {len(val_keys)}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((192, 96))])

    # ---- inference over both splits ----
    val_recs = collect_split(val_keys, val_seqs, mc_model, transform, "val")
    test_recs = collect_split(test_keys, test_seqs, mc_model, transform, "test")

    # ---- UA-EMA calibration bounds from validation ----
    val_u = np.concatenate([r['u20'] for r in val_recs])
    u_lo, u_hi = float(np.percentile(val_u, 5)), float(np.percentile(val_u, 95))
    print(f"UA-EMA calibration bounds (val 5th/95th pct): {u_lo:.5f} / {u_hi:.5f}")

    # ---- Kalman grid search on validation ----
    def grid_search(filt, grid_q, grid_r):
        rows = []
        for Q in grid_q:
            for R in grid_r:
                ss, jj = [], []
                for rec in val_recs:
                    fp = filt(rec['mean20'], Q, R)
                    ss.append(seq_ssim(fp, rec['gt']))
                    jj.append(compute_jitter(fp))
                rows.append({'Q': Q, 'R': R, 'ssim': float(np.mean(ss)),
                             'jitter': float(np.mean(jj))})
        return rows

    print("Grid search: random-walk KF ...")
    rw_grid = grid_search(filt_kalman_rw, KF_RW_GRID_Q, KF_RW_GRID_R)
    print("Grid search: constant-velocity KF ...")
    cv_grid = grid_search(filt_kalman_cv, KF_CV_GRID_Q, KF_CV_GRID_R)

    # Selection: (a) max SSIM; (b) max SSIM subject to jitter <= UA-EMA val jitter
    ua_val_jit = float(np.mean([compute_jitter(
        filt_ema(r['mean20'], ua_ema_alphas(r['u20'], u_lo, u_hi))) for r in val_recs]))

    def select(rows):
        best_ssim = max(rows, key=lambda r: r['ssim'])
        constrained = [r for r in rows if r['jitter'] <= ua_val_jit]
        best_con = max(constrained, key=lambda r: r['ssim']) if constrained else None
        return best_ssim, best_con

    rw_best, rw_best_con = select(rw_grid)
    cv_best, cv_best_con = select(cv_grid)
    print(f"RW-KF best (max SSIM): {rw_best}; constrained: {rw_best_con}")
    print(f"CV-KF best (max SSIM): {cv_best}; constrained: {cv_best_con}")

    # Use the comparable-stability operating point (jitter <= UA-EMA's) as the
    # headline tuned KF; fall back to max-SSIM if the constraint is infeasible.
    rw_sel = rw_best_con or rw_best
    cv_sel = cv_best_con or cv_best

    # ---- evaluate all filters on the test split ----
    methods = evaluate_filters(test_recs, (u_lo, u_hi),
                               (rw_sel['Q'], rw_sel['R']), (cv_sel['Q'], cv_sel['R']))

    raw_jit = np.mean(methods['no_filter']['jitter'])
    table2 = {}
    for name, d in methods.items():
        table2[name] = {
            'ssim': float(np.mean(d['ssim'])),
            'ssim_std': float(np.std(d['ssim'])),
            'jitter': float(np.mean(d['jitter'])),
            'jitter_std': float(np.std(d['jitter'])),
            'jitter_reduction_pct': float((1 - np.mean(d['jitter']) / raw_jit) * 100),
        }
        print(f"{name:<12} SSIM={table2[name]['ssim']:.4f} "
              f"jit={table2[name]['jitter']:.6f} red={table2[name]['jitter_reduction_pct']:.1f}%")

    # ---- paired significance: UA-EMA vs each competitor (Holm-corrected) ----
    comparisons = {}
    pvals = []
    for other in ['kalman_rw', 'kalman_cv', 'fixed_ema', 'moving_avg']:
        for metric in ['ssim', 'jitter']:
            a = np.array(methods['ua_ema'][metric])
            b = np.array(methods[other][metric])
            try:
                w = sstats.wilcoxon(a, b)
                p = float(w.pvalue)
            except ValueError:
                p = 1.0
            comparisons[f'ua_ema_vs_{other}_{metric}'] = {
                'mean_diff': float(np.mean(a - b)), 'p_raw': p}
            pvals.append((f'ua_ema_vs_{other}_{metric}', p))
    # Holm correction
    pvals.sort(key=lambda x: x[1])
    m = len(pvals)
    for rank, (name, p) in enumerate(pvals):
        p_holm = min(1.0, max((m - r) * pv for r, (_, pv) in enumerate(pvals[:rank + 1])))
        comparisons[name]['p_holm'] = float(p_holm)

    # ---- N-pass sweep (accuracy/stability of UA-EMA with N=5/10/20) ----
    n_sweep = {}
    for n_sub, mk, uk in ((5, 'mean5', 'u5'), (10, 'mean10', 'u10'), (20, 'mean20', 'u20')):
        val_u_n = np.concatenate([r[uk] for r in val_recs])
        lo, hi = float(np.percentile(val_u_n, 5)), float(np.percentile(val_u_n, 95))
        ss, jj, ssr, jjr = [], [], [], []
        for rec in test_recs:
            fp = filt_ema(rec[mk], ua_ema_alphas(rec[uk], lo, hi))
            ss.append(seq_ssim(fp, rec['gt']))
            jj.append(compute_jitter(fp))
            ssr.append(seq_ssim(rec[mk], rec['gt']))
            jjr.append(compute_jitter(rec[mk]))
        n_sweep[f'N{n_sub}'] = {
            'ua_ema_ssim': float(np.mean(ss)), 'ua_ema_jitter': float(np.mean(jj)),
            'raw_ssim': float(np.mean(ssr)), 'raw_jitter': float(np.mean(jjr)),
            'u_bounds': [lo, hi],
        }
        print(f"N={n_sub}: UA-EMA SSIM={np.mean(ss):.4f} jitter={np.mean(jj):.6f}")

    # ---- latency benchmark on this GPU ----
    ir_tensor, _ = load_sample(*[p for p in [test_seqs[test_keys[0]][0][1],
                                             test_seqs[test_keys[0]][0][2]]], transform)
    ir_tensor = ir_tensor.unsqueeze(0).to(device)
    latency = {}
    with torch.no_grad():
        for _ in range(5):
            base_model(ir_tensor)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            base_model(ir_tensor)
        torch.cuda.synchronize()
        latency['base_ms'] = (time.time() - t) / 50 * 1000
    for n in (5, 10, 20):
        mc_forward(mc_model, ir_tensor, n)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            mc_forward(mc_model, ir_tensor, n)
        torch.cuda.synchronize()
        latency[f'mc{n}_ms'] = (time.time() - t) / 20 * 1000
    latency['gpu'] = torch.cuda.get_device_name(0)
    print("Latency:", {k: (f"{v:.1f}" if isinstance(v, float) else v) for k, v in latency.items()})

    # ---- calibration analysis (pixel-level reliability + ENCE) ----
    sig_all, err_all = [], []
    frame_u, frame_rmse = [], []
    for rec in test_recs:
        for mp, sg, gt, u in zip(rec['mean20'], rec['sig20'], rec['gt'], rec['u20']):
            err = np.abs(np.clip(mp, 0, 1) - gt)
            sig_all.append(sg[::4, ::4].ravel())   # subsample pixels 1/16
            err_all.append(err[::4, ::4].ravel())
            frame_u.append(u)
            frame_rmse.append(float(np.sqrt(np.mean((np.clip(mp, 0, 1) - gt) ** 2))))
    sig_all = np.concatenate(sig_all)
    err_all = np.concatenate(err_all)
    order = np.argsort(sig_all)
    n_bins = 10
    bins = np.array_split(order, n_bins)
    rel_sigma = [float(np.mean(sig_all[b])) for b in bins]
    rel_rmse = [float(np.sqrt(np.mean(err_all[b] ** 2))) for b in bins]
    ence = float(np.mean([abs(r - s) / max(s, 1e-8) for s, r in zip(rel_sigma, rel_rmse)]))
    rho_pix = float(sstats.spearmanr(sig_all[::50], err_all[::50]).statistic)
    rho_frame = float(sstats.spearmanr(frame_u, frame_rmse).statistic)
    print(f"ENCE={ence:.3f}  Spearman(pixel)={rho_pix:.3f}  Spearman(frame)={rho_frame:.3f}")

    # ---- Table 1/3 replication check (base model, 300/condition, as in 16) ----
    test_by_cond = {'uncover': [], 'cover1': [], 'cover2': []}
    for ir, pm, k in data:
        if k[0] in test_subjects and k[1] in test_by_cond:
            test_by_cond[k[1]].append((ir, pm))
    table3 = {}
    for cond in test_by_cond:
        mets = []
        for ir_path, pm_path in tqdm(test_by_cond[cond][:300], desc=f"t13-{cond}"):
            ir_t, gt = load_sample(ir_path, pm_path, transform)
            with torch.no_grad():
                pred = base_model(ir_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            pred = np.clip(cv2.resize(pred, (gt.shape[1], gt.shape[0])), 0, 1)
            mets.append(ssim(pred, gt, data_range=1.0))
        table3[cond] = {'ssim_mean': float(np.mean(mets)), 'ssim_std': float(np.std(mets)),
                        'n': len(mets)}
        print(f"table3 {cond}: SSIM {np.mean(mets):.4f}")

    results = {
        'meta': {
            'script': '24_revision_temporal_calibration.py',
            'git_commit': commit, 'config_hash': config_hash, 'seed': SEED,
            'timestamp': datetime.now().isoformat(), 'device': latency['gpu'],
            'config': config,
        },
        'dataset_stats': dataset_stats,
        'ua_ema_calibration_bounds_val_5_95pct': [u_lo, u_hi],
        'kalman_rw': {'grid': rw_grid, 'best_ssim': rw_best,
                      'best_constrained': rw_best_con, 'selected': rw_sel},
        'kalman_cv': {'grid': cv_grid, 'best_ssim': cv_best,
                      'best_constrained': cv_best_con, 'selected': cv_sel},
        'table2_revised': table2,
        'significance': comparisons,
        'n_pass_sweep': n_sweep,
        'latency_ms': latency,
        'calibration': {'ence': ence, 'spearman_pixel': rho_pix,
                        'spearman_frame': rho_frame,
                        'reliability_sigma': rel_sigma, 'reliability_rmse': rel_rmse},
        'table3_check': table3,
        'runtime_min': (time.time() - t0) / 60,
    }
    out = RESULT_DIR / "temporal_revision.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}  ({results['runtime_min']:.1f} min)")


if __name__ == "__main__":
    main()

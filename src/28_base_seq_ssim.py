#!/usr/bin/env python3
"""Single-pass (no dropout) base model on the same 30 test sequences —
completes the revised ablation table. Output: result/revision/base_seq.json"""
import sys
sys.path.insert(0, '/home/wjeong/cc/test/revision')
import importlib
m = importlib.import_module('24_revision_temporal_calibration')
import numpy as np, torch, json, cv2
from torchvision import transforms
import segmentation_models_pytorch as smp

model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
ckpt = torch.load(m.MODEL_DIR / "U-Net(ResNet50)_E50.pth", map_location=m.device)
sd = {k: v for k, v in ckpt['model_state_dict'].items() if not k.endswith(('total_ops','total_params'))}
model.load_state_dict(sd, strict=False); model = model.to(m.device).eval()

data = m.load_data()
subs = list(set(k[0] for _,_,k in data)); np.random.seed(m.SEED); np.random.shuffle(subs)
test_subjects = set(subs[:len(subs)//5])
seqs = {}
for ir, pm, k in data:
    if k[0] in test_subjects:
        seqs.setdefault((k[0],k[1]), []).append((k[2], ir, pm))
for sk in seqs: seqs[sk].sort(key=lambda x: x[0])
keys = [k for k in seqs if len(seqs[k]) >= 20][:m.N_SEQ]
tr = transforms.Compose([transforms.ToTensor(), transforms.Resize((192,96))])
ss, jj = [], []
for sk in keys:
    preds, gts = [], []
    for _, irp, pmp in seqs[sk][:m.SEQ_LEN]:
        irt, gt = m.load_sample(irp, pmp, tr)
        with torch.no_grad():
            p = model(irt.unsqueeze(0).to(m.device)).squeeze().cpu().numpy()
        preds.append(cv2.resize(p, (gt.shape[1], gt.shape[0]))); gts.append(gt)
    ss.append(m.seq_ssim(preds, gts)); jj.append(m.compute_jitter(preds))
out = {'base_singlepass_seq_ssim': float(np.mean(ss)), 'base_singlepass_seq_jitter': float(np.mean(jj)), 'n_seq': len(keys)}
print(out)
json.dump(out, open(m.RESULT_DIR / "base_seq.json", 'w'), indent=2)

"""
Tecto Seizure v2 — Force r-gate malfunction.
If r silently sits at 0.5, it's protective. What if we force it open/closed?
Biological analogy: thalamic reticular nucleus failure in photosensitive epilepsy.

Usage: python seizure_v2.py
"""
import numpy as np, cv2, torch, json, math, csv
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Paths resolved relative to this file's location (portable)
SRC_DIR = Path(__file__).parent
_ROOT = SRC_DIR.parent
_DATA = _ROOT / 'data'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIM = 384; POOL_H, POOL_W = 4, 8; N_SPATIAL = 32; MICRO_H2 = 8; MICRO_OUT = 256; H = 128
GRU_IN = 384+384+384+256
AX = 36; AY0, AY1 = 0, 30; AZ = 60
EYE_OFFSET = 3.0; EYE_ANGLE = math.radians(25.0)
FOOD_R = 5.0

OUT_DIR = _ROOT / 'seizure_test'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading DINOv2...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
dino_model = AutoModel.from_pretrained('facebook/dinov2-small').to(DEVICE).eval()
with open(_DATA / 'best_brain_v8.json') as f:
    _mic = json.load(f)['micro']
FW1 = np.array(_mic['W1']); Fb1 = np.array(_mic['b1'])
FW2 = np.array(_mic['W2']); Fb2 = np.array(_mic['b2'])

def micro_fwd(pd):
    N = len(pd); pf = pd.reshape(N*N_SPATIAL, DIM)
    return (np.maximum(0, pf@FW1.T+Fb1)@FW2.T+Fb2).reshape(N, N_SPATIAL, MICRO_H2)

def retina1408(L_img, R_img):
    L = [Image.fromarray(L_img)]; R = [Image.fromarray(R_img)]
    iL = processor(images=L, return_tensors="pt").to(DEVICE)
    iR = processor(images=R, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        oL = dino_model(**iL); oR = dino_model(**iR)
    cL = oL.last_hidden_state[:, 0, :].cpu().numpy()[0]; cL /= np.linalg.norm(cL) + 1e-10
    cR = oR.last_hidden_state[:, 0, :].cpu().numpy()[0]; cR /= np.linalg.norm(cR) + 1e-10
    disp = cL - cR
    pLt = oL.last_hidden_state[:, 1:, :].cpu().numpy()[0]
    pRt = oR.last_hidden_state[:, 1:, :].cpu().numpy()[0]
    pLt /= np.linalg.norm(pLt, axis=1, keepdims=True) + 1e-10
    pRt /= np.linalg.norm(pRt, axis=1, keepdims=True) + 1e-10
    pd = np.zeros((N_SPATIAL, DIM), dtype=np.float32)
    for ph in range(POOL_H):
        for pw in range(POOL_W):
            pd[ph*POOL_W+pw, :] = (pLt-pRt)[ph*4*8+pw*2:ph*4*8+pw*2+8, :].mean(axis=0)
    ms = micro_fwd(pd.reshape(1, N_SPATIAL, DIM)).reshape(1, MICRO_OUT)
    return np.concatenate([cL, cR, disp, ms.flatten()]).astype(np.float32).reshape(1, GRU_IN)

def rot(dx, dz, a):
    return dx*math.cos(a)-dz*math.sin(a), dx*math.sin(a)+dz*math.cos(a)

def render_lateral(fx, fy, fz, fh, foods):
    sz = (280, 280); cx, cy = sz[1]//2, sz[0]//2; fl = 80
    L_ang = fh - EYE_ANGLE; R_ang = fh + EYE_ANGLE
    L_ex = fx - EYE_OFFSET*math.cos(fh); L_ez = fz + EYE_OFFSET*math.sin(fh)
    R_ex = fx + EYE_OFFSET*math.cos(fh); R_ez = fz - EYE_OFFSET*math.sin(fh)
    imgs = {}
    for ex, ez, eh, lbl in [(L_ex, L_ez, L_ang, 'L'), (R_ex, R_ez, R_ang, 'R')]:
        img = np.zeros((*sz, 3), np.uint8)
        for py in range(sz[0]):
            img[py, :] = [int(60+80*py/sz[0]), int((60+80*py/sz[0])*0.7), 40]
        for ffx, ffy, ffz, fr in foods:
            rlx, rlz = rot(ffx-ex, ffz-ez, eh); rly = ffy - fy
            d = math.sqrt(rlx**2 + rly**2 + rlz**2)
            if d >= 0.5 and rlz > 0.3:
                px = int(cx + fl*rlx/max(rlz, 0.5))
                py = int(cy + fl*rly/max(rlz, 0.5))
                pr = max(3, int(fl*fr/max(rlz, 0.5)))
                if 0 <= px < sz[1] and 0 < py < sz[0]:
                    cv2.circle(img, (px, py), pr, (0, 255, 0), -1)
        imgs[lbl] = img
    return imgs['L'], imgs['R']

class SeizureBrain:
    """GRU where we can FORCE r to malfunction."""
    def __init__(self, H=128):
        self.H = H
        self.W_z = np.zeros((H, GRU_IN)); self.U_z = np.zeros((H, H)); self.b_z = np.zeros(H)
        self.W_r = np.zeros((H, GRU_IN)); self.U_r = np.zeros((H, H)); self.b_r = np.zeros(H)
        self.W_h = np.zeros((H, GRU_IN)); self.U_h = np.zeros((H, H)); self.b_h = np.zeros(H)
        self.W1 = np.zeros((32, H)); self.b1 = np.zeros(32)
        self.W2 = np.zeros((4, 32)); self.b2 = np.zeros(4)

    def set_params(self, flat):
        H = self.H; idx = 0
        for g in ['z', 'r', 'h']:
            for p in ['W', 'U', 'b']:
                a = getattr(self, f'{p}_{g}'); m = a.size
                a.flat = flat[idx:idx+m]; idx += m
        for a in [self.W1, self.b1, self.W2, self.b2]:
            m = a.size; a.flat = flat[idx:idx+m]; idx += m

    def forward(self, x, h, force_r=None):
        """force_r: if not None, override r gate with this value (0-1 float or array)."""
        H = self.H
        z = 1/(1+np.exp(-np.clip(x@self.W_z.T + h@self.U_z.T + self.b_z, -10, 10)))
        r = 1/(1+np.exp(-np.clip(x@self.W_r.T + h@self.U_r.T + self.b_r, -10, 10)))
        if force_r is not None:
            r = np.ones_like(r) * force_r  # FORCE the gate
        ht_ = np.tanh(x@self.W_h.T + (r*h)@self.U_h.T + self.b_h)
        hn = (1-z)*h + z*ht_; hn *= 0.999
        return np.tanh(np.maximum(0, hn@self.W1.T+self.b1)@self.W2.T+self.b2), hn, z, r, ht_

# ============================================================
BRAIN_PATH = _DATA / 'v12_mixed_H128.npy'
bf = np.load(BRAIN_PATH)
print(f"Brain loaded.")

FISH_X, FISH_Y, FISH_Z = 0, 15, 20
FOOD_X, FOOD_Y, FOOD_Z = 0, 16, 28
EPISODE_STEPS = 500

# r-malfunction patterns
R_CONDITIONS = [
    ("r_normal", None, "r = trained (always 0.5) — control"),
    ("r_forced_zero", 0.0, "r forced to 0 — total amnesia every step"),
    ("r_forced_one", 1.0, "r forced to 1 — cannot forget, past dominates"),
    ("r_osc_2Hz", 'osc_2', "r oscillates 0↔1 at 2 Hz — thalamic flicker"),
    ("r_osc_8Hz", 'osc_8', "r oscillates 0↔1 at 8 Hz — seizure range"),
    ("r_osc_16Hz", 'osc_16', "r oscillates 0↔1 at 16 Hz — rapid seizure"),
    ("r_random", 'rand', "r random 0↔1 each step — gate chaos"),
]

# Food flicker: fixed at 0 Hz (food always visible) OR at a resonant frequency
FOOD_FLICKER = [(False, "static food"),
                (True, "food flicker 8 Hz")]

results = []

for r_name, r_mode, r_desc in R_CONDITIONS:
    for flicker, flicker_desc in FOOD_FLICKER:
        brain = SeizureBrain()
        brain.set_params(bf)
        ht = np.zeros((1, brain.H))
        h_history, out_history, z_history, r_history = [], [], [], []
        food_visible = True
        flicker_interval = max(1, int(round(1.0 / 8))) if flicker else 99999  # 8 Hz if flickering

        for st in range(EPISODE_STEPS):
            if flicker and st % flicker_interval == 0:
                food_visible = not food_visible

            foods = [(FOOD_X, FOOD_Y, FOOD_Z, FOOD_R)] if food_visible else []
            L, R = render_lateral(FISH_X, FISH_Y, FISH_Z, 0.0, foods)
            enc = retina1408(L, R)

            # Compute force_r value
            if r_mode == 'osc_2':
                force_r = float(st % max(1, int(round(1.0/2))) < max(1, int(round(1.0/2)))//2)
            elif r_mode == 'osc_8':
                force_r = float(st % max(1, int(round(1.0/8))) < max(1, int(round(1.0/8)))//2)
            elif r_mode == 'osc_16':
                force_r = float(st % max(1, int(round(1.0/16))) < max(1, int(round(1.0/16)))//2)
            elif r_mode == 'rand':
                force_r = float(np.random.random() > 0.5)
            else:
                force_r = r_mode  # None or float

            out, ht, z, r, ht_ = brain.forward(enc, ht, force_r=force_r)
            h_norm = float(np.linalg.norm(ht))
            h_history.append(h_norm)
            out_history.append([float(t) for t in out[0]])
            z_history.append(float(np.mean(z)))
            r_history.append(float(np.mean(r)))

        # Metrics
        h_vals = np.array(h_history[50:])  # skip initial transient
        h_mean = float(np.mean(h_vals))
        h_std = float(np.std(h_vals))
        h_max = float(np.max(h_vals))
        h_osc_amp = h_max - h_mean

        # Autocorrelation for oscillation period
        h_detrend = h_vals - h_mean
        ac = np.correlate(h_detrend, h_detrend, mode='full')
        ac = ac[len(ac)//2:] / (ac[len(ac)//2] + 1e-10)
        peaks = [i for i in range(3, len(ac)-1) if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.3]
        osc_period = peaks[0] if peaks else 0

        # Turn instability
        L_vals = np.array([o[0] for o in out_history[50:]])
        R_vals = np.array([o[1] for o in out_history[50:]])
        turns = (R_vals - L_vals)
        turn_std = float(np.std(turns))
        turn_range = float(np.max(turns) - np.min(turns))

        # z gate: does it still function?
        z_vals = np.array(z_history[50:])
        z_final = float(np.mean(z_vals[-50:]))
        z_std = float(np.std(z_vals))

        # r gate: actual effective values
        r_vals = np.array(r_history[50:])
        r_final = float(np.mean(r_vals[-50:]))
        r_std_val = float(np.std(r_vals))

        # Is this seizure-like?
        seizure = (h_osc_amp > 2.0 * h_std and osc_period > 0) or (turn_std > 0.3)

        results.append({
            'r_condition': r_name, 'flicker': flicker_desc,
            'h_mean': h_mean, 'h_std': h_std, 'h_max': h_max,
            'h_osc_amp': h_osc_amp, 'osc_period': osc_period,
            'turn_std': turn_std, 'turn_range': turn_range,
            'z_final': z_final, 'z_std': z_std,
            'r_final': r_final, 'r_std': r_std_val,
            'seizure': seizure,
        })

        label = "SEIZURE!" if seizure else "normal"
        print(f"  {r_name:18s} + {flicker_desc:20s} | "
              f"h_osc={h_osc_amp:.4f}  osc_p={osc_period:>4d}  "
              f"turn_std={turn_std:.4f}  turn_range={turn_range:.3f}  "
              f"z={z_final:.4f}±{z_std:.4f}  r={r_final:.3f}  [{label}]")

# ============================================================
print(f"\n{'='*80}")
print("R-GATE MALFUNCTION SUMMARY")
print(f"{'='*80}")
print(f"{'Condition':<25s} {'Flicker':<22s} {'h_osc':>8s} {'turn_std':>9s} {'turn_rng':>9s} {'z_final':>8s} {'Status'}")
print("-" * 85)
for r in results:
    status = "SEIZURE" if r['seizure'] else "normal"
    print(f"  {r['r_condition']:<23s} {r['flicker']:<20s} "
          f"{r['h_osc_amp']:8.4f} {r['turn_std']:9.4f} {r['turn_range']:9.4f} "
          f"{r['z_final']:8.4f}  {status}")

# Key comparison
print(f"\n  --- Protective effect of r silence ---")
normal = [r for r in results if r['r_condition'] == 'r_normal' and 'static' in r['flicker']][0]
for r in results:
    if r['r_condition'] != 'r_normal':
        ratio = r['h_osc_amp'] / max(0.001, normal['h_osc_amp'])
        ratio_t = r['turn_std'] / max(0.001, normal['turn_std'])
        print(f"  {r['r_condition']:18s} + {r['flicker']:20s}: "
              f"h_osc={ratio:.1f}x normal  turn_std={ratio_t:.1f}x normal  "
              f"seizure={r['seizure']}")

csv_path = OUT_DIR / 'seizure_v2_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"\nSaved: {csv_path}")
print("DONE")

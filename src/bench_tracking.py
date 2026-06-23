"""
Food tracking benchmark — 500 fish, 1 random-walk food, pure torus.
Measures: Qualified ATE rate (fish that see food at spawn, standard ATE metric).
V12 brain. Pure generalization test — no dynamic-food training.

Usage: python bench_tracking.py [brain.npy] [n_fish] [n_steps]
"""
import numpy as np, cv2, torch, json, math, time, sys
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

OUT_DIR = _ROOT / 'benchmarks'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading DINOv2...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
dino = AutoModel.from_pretrained('facebook/dinov2-small').to(DEVICE).eval()
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
        oL = dino(**iL); oR = dino(**iR)
    cL = oL.last_hidden_state[:, 0, :].cpu().numpy()[0]
    cL /= np.linalg.norm(cL) + 1e-10
    cR = oR.last_hidden_state[:, 0, :].cpu().numpy()[0]
    cR /= np.linalg.norm(cR) + 1e-10
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

def food_visible(fx, fy, fz, fh, ffx, ffy, ffz):
    L_ang = fh - EYE_ANGLE; L_ex = fx - EYE_OFFSET*math.cos(fh); L_ez = fz + EYE_OFFSET*math.sin(fh)
    R_ang = fh + EYE_ANGLE; R_ex = fx + EYE_OFFSET*math.cos(fh); R_ez = fz - EYE_OFFSET*math.sin(fh)
    FL = 80; SZ_W = 280; SZ_H = 280; cx = SZ_W//2; cy = SZ_H//2
    for ex, ez, eh in [(L_ex, L_ez, L_ang), (R_ex, R_ez, R_ang)]:
        rlx, rlz = rot(ffx-ex, ffz-ez, eh); rly = ffy - fy
        if rlz > 0.3:
            px = int(cx + FL*rlx/max(rlz, 0.5))
            py = int(cy + FL*rly/max(rlz, 0.5))
            if 0 <= px < SZ_W and 0 < py < SZ_H:
                return True
    return False

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

class GRU:
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

    def forward(self, x, h):
        H = self.H
        z = 1/(1+np.exp(-np.clip(x@self.W_z.T + h@self.U_z.T + self.b_z, -10, 10)))
        r = 1/(1+np.exp(-np.clip(x@self.W_r.T + h@self.U_r.T + self.b_r, -10, 10)))
        ht_ = np.tanh(x@self.W_h.T + (r*h)@self.U_h.T + self.b_h)
        hn = (1-z)*h + z*ht_; hn *= 0.999
        return np.tanh(np.maximum(0, hn@self.W1.T+self.b1)@self.W2.T+self.b2), hn

# ============================================================
# Load brain
# ============================================================
brain_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    _DATA / 'v12_mixed_H128.npy'
n_fish = int(sys.argv[2]) if len(sys.argv) > 2 else 500
n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400

bf = np.load(brain_path)
brain = GRU(); brain.set_params(bf)
print(f"Brain: {brain_path.name} ({len(bf)}p, H={brain.H})")
print(f"N={n_fish} fish, {n_steps} max steps")
print()

FOOD_R = 5.0

rng = np.random.RandomState(42)
results = []

for fi in range(n_fish):
    # ---- spawn fish ----
    fx = rng.uniform(-AX*0.9, AX*0.9)
    fy = rng.uniform(AY0+3, AY1-3)
    fz = rng.uniform(5, AZ-5)
    fh = rng.uniform(-math.pi, math.pi)
    ht = np.zeros((1, brain.H))

    # ---- spawn food (random walk with heading) ----
    ffx = rng.uniform(-AX*0.8, AX*0.8)
    ffy = rng.uniform(AY0+3, AY1-3)
    ffz = rng.uniform(5, AZ-5)
    ffh = rng.uniform(-math.pi, math.pi)  # food heading for random walk

    # Check if fish qualifies
    qualified = food_visible(fx, fy, fz, fh, ffx, ffy, ffz)

    ate = False; steps_survived = 0
    start_dist = math.sqrt((fx-ffx)**2 + (fy-ffy)**2 + (fz-ffz)**2)
    end_dist = start_dist
    min_dist = start_dist

    for st in range(n_steps):
        # ---- move food (random walk, speed 0.5-2.5 units/step) ----
        ffh += rng.uniform(-0.3, 0.3)  # smooth heading drift
        food_speed = rng.uniform(0.5, 2.5)
        ffx += food_speed * math.sin(ffh)
        ffz += food_speed * math.cos(ffh)
        ffy += rng.uniform(-0.3, 0.3)
        ffy = np.clip(ffy, AY0+1, AY1-1)

        # Torus wrap food
        while ffx > AX: ffx -= 2*AX
        while ffx < -AX: ffx += 2*AX
        while ffz > AZ: ffz -= AZ
        while ffz < 0: ffz += AZ

        # ---- check termination ----
        out_of_bounds = abs(fx) > AX or fz < 0 or fz > AZ or fy < AY0+0.5 or fy > AY1-0.5
        dist = math.sqrt((fx-ffx)**2 + (fy-ffy)**2 + (fz-ffz)**2)
        min_dist = min(min_dist, dist)
        just_ate = dist < FOOD_R + 2

        # ---- vision + brain ----
        L, R = render_lateral(fx, fy, fz, fh, [(ffx, ffy, ffz, FOOD_R)])
        enc = retina1408(L, R)
        out, ht = brain.forward(enc, ht)
        lt, rt, ut, dt = float(out[0,0]), float(out[0,1]), float(out[0,2]), float(out[0,3])

        # ---- move fish ----
        fwd = (lt+rt)/2 * 8
        fh += (rt-lt) * 0.5
        fx = fx + fwd*math.sin(fh)
        fz = fz + fwd*math.cos(fh)
        fy = fy + (ut-dt) * 5

        end_dist = dist
        steps_survived = st + 1

        if just_ate:
            ate = True
            break
        if out_of_bounds:
            break

    results.append({
        'idx': fi,
        'qualified': qualified,
        'ate': ate,
        'start_dist': round(start_dist, 1),
        'end_dist': round(end_dist, 1),
        'min_dist': round(min_dist, 1),
        'steps': steps_survived,
    })

    if (fi+1) % 50 == 0 or fi == 0:
        q_so_far = [r for r in results if r['qualified']]
        ate_so_far = sum(1 for r in q_so_far if r['ate'])
        all_ate = sum(1 for r in results if r['ate'])
        print(f"  [{fi+1:>4d}/{n_fish}] Q:{ate_so_far}/{len(q_so_far)}={ate_so_far/max(1,len(q_so_far)):.2f}  "
              f"All:{all_ate}/{fi+1}={all_ate/(fi+1):.2f}")

# ============================================================
# Results
# ============================================================
qualified_results = [r for r in results if r['qualified']]
unqualified_results = [r for r in results if not r['qualified']]
q_ate = sum(1 for r in qualified_results if r['ate'])
all_ate = sum(1 for r in results if r['ate'])

print(f"\n{'='*70}")
print(f"BENCHMARK: Food Tracking (random walk)")
print(f"  Brain: {brain_path.name}  N={n_fish}  MaxSteps={n_steps}")
print(f"{'='*70}")
print(f"  Total fish:       {n_fish}")
print(f"  Qualified (saw food at spawn):  {len(qualified_results)} ({len(qualified_results)/n_fish*100:.1f}%)")
print(f"  Unqualified (didn't see food):  {len(unqualified_results)}")
print()
print(f"  Qualified ATE rate:  {q_ate}/{len(qualified_results)} = {q_ate/max(1,len(qualified_results)):.3f}")
print(f"  All-fish ATE rate:   {all_ate}/{n_fish} = {all_ate/n_fish:.3f}")
print()

# Detailed breakdown
print(f"  --- Qualified fish breakdown ---")
if qualified_results:
    q_ate_list = [r for r in qualified_results if r['ate']]
    q_miss_list = [r for r in qualified_results if not r['ate']]
    print(f"  ATE! ({len(q_ate_list)}):  "
          f"mean_steps={np.mean([r['steps'] for r in q_ate_list]):.1f}  "
          f"mean_start_dist={np.mean([r['start_dist'] for r in q_ate_list]):.1f}")
    if q_miss_list:
        print(f"  Miss ({len(q_miss_list)}):  "
              f"mean_steps={np.mean([r['steps'] for r in q_miss_list]):.1f}  "
              f"mean_start_dist={np.mean([r['start_dist'] for r in q_miss_list]):.1f}  "
              f"mean_min_dist={np.mean([r['min_dist'] for r in q_miss_list]):.1f}")
    else:
        print(f"  Miss: 0 — PERFECT! Every qualified fish ate.")

print(f"\n  --- Unqualified fish breakdown ---")
if unqualified_results:
    u_ate_list = [r for r in unqualified_results if r['ate']]
    print(f"  Got lucky (ate anyway): {len(u_ate_list)}")
    print(f"  Mean steps: {np.mean([r['steps'] for r in unqualified_results]):.1f}")
    print(f"  Mean end dist: {np.mean([r['end_dist'] for r in unqualified_results]):.1f}")

# Save CSV
import csv
csv_path = OUT_DIR / f'tracking_bench_{brain_path.stem}.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['idx','qualified','ate','start_dist','end_dist','min_dist','steps'])
    w.writeheader()
    for r in results:
        w.writerow(r)
print(f"\nSaved: {csv_path}")
print("DONE")

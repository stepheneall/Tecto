"""
Phase 4: Ablation Verification
Zero out top-weight hidden dims vs bottom vs random.
Measure ATE rate and collision rate in mixed scenarios (food + 3 obstacles).

Usage: python ablation_verify.py
"""
import numpy as np, cv2, torch, json, math, csv, copy
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
FOOD_R = 5.0; FISH_BODY_R = 1.5
N_TRIALS = 150; N_STEPS = 300; N_OBS = 3

OUT_DIR = _ROOT / 'circuit_analysis'
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

def render_lateral(fx, fy, fz, fh, foods, obstacles):
    sz = (280, 280); cx, cy = sz[1]//2, sz[0]//2; fl = 80
    L_ang = fh - EYE_ANGLE; R_ang = fh + EYE_ANGLE
    L_ex = fx - EYE_OFFSET*math.cos(fh); L_ez = fz + EYE_OFFSET*math.sin(fh)
    R_ex = fx + EYE_OFFSET*math.cos(fh); R_ez = fz - EYE_OFFSET*math.sin(fh)
    imgs = {}
    for ex, ez, eh, lbl in [(L_ex, L_ez, L_ang, 'L'), (R_ex, R_ez, R_ang, 'R')]:
        img = np.zeros((*sz, 3), np.uint8)
        for py in range(sz[0]):
            img[py, :] = [int(60+80*py/sz[0]), int((60+80*py/sz[0])*0.7), 40]
        for ox, oy, oz, orx, ory, orz in obstacles:
            rlx, rlz = rot(ox-ex, oz-ez, eh); rly = oy - fy
            d = math.sqrt(rlx**2 + rly**2 + rlz**2)
            if d >= 0.5 and rlz > 0.3:
                px = int(cx + fl*rlx/max(rlz, 0.5))
                py = int(cy + fl*rly/max(rlz, 0.5))
                prx = max(2, int(fl*orx/max(rlz, 0.5)))
                pry = max(2, int(fl*ory/max(rlz, 0.5)))
                cv2.rectangle(img, (max(0, px-prx), max(0, py-pry)),
                              (min(sz[1]-1, px+prx), min(sz[0]-1, py+pry)), (80, 80, 80), -1)
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

class AblatableGRU:
    """GRU where we can zero out specific hidden dimensions after h_new."""
    def __init__(self, H=128):
        self.H = H
        self.W_z = np.zeros((H, GRU_IN)); self.U_z = np.zeros((H, H)); self.b_z = np.zeros(H)
        self.W_r = np.zeros((H, GRU_IN)); self.U_r = np.zeros((H, H)); self.b_r = np.zeros(H)
        self.W_h = np.zeros((H, GRU_IN)); self.U_h = np.zeros((H, H)); self.b_h = np.zeros(H)
        self.W1 = np.zeros((32, H)); self.b1 = np.zeros(32)
        self.W2 = np.zeros((4, 32)); self.b2 = np.zeros(4)
        self.ablate_mask = np.ones(H)  # 1 = keep, 0 = zero out

    def set_params(self, flat):
        H = self.H; idx = 0
        for g in ['z', 'r', 'h']:
            for p in ['W', 'U', 'b']:
                a = getattr(self, f'{p}_{g}'); m = a.size
                a.flat = flat[idx:idx+m]; idx += m
        for a in [self.W1, self.b1, self.W2, self.b2]:
            m = a.size; a.flat = flat[idx:idx+m]; idx += m

    def set_ablation(self, dims_to_zero):
        self.ablate_mask = np.ones(self.H)
        for d in dims_to_zero:
            if 0 <= d < self.H:
                self.ablate_mask[d] = 0

    def forward(self, x, h):
        H = self.H
        z = 1/(1+np.exp(-np.clip(x@self.W_z.T + h@self.U_z.T + self.b_z, -10, 10)))
        r = 1/(1+np.exp(-np.clip(x@self.W_r.T + h@self.U_r.T + self.b_r, -10, 10)))
        ht_ = np.tanh(x@self.W_h.T + (r*h)@self.U_h.T + self.b_h)
        hn = (1-z)*h + z*ht_; hn *= 0.999
        # Ablation: zero out specific dims
        hn = hn * self.ablate_mask.reshape(1, -1)
        return np.tanh(np.maximum(0, hn@self.W1.T+self.b1)@self.W2.T+self.b2), hn

# ============================================================
# Load brain and compute dim importance
# ============================================================
BRAIN_PATH = _DATA / 'v12_mixed_H128.npy'
bf = np.load(BRAIN_PATH)

# Compute dim importance from weight magnitude
def compute_importance(bf_data):
    """Load weights and compute W_eff magnitude per dim."""
    idx = 0; H = 128
    w1 = np.zeros((32, H)); w2 = np.zeros((4, 32))
    # Skip GRU weights
    for g in ['z', 'r', 'h']:
        for p in ['W', 'U', 'b']:
            if p == 'W': s = H * GRU_IN
            elif p == 'U': s = H * H
            else: s = H
            idx += s
    w1_s = w1.size; w1.flat = bf_data[idx:idx+w1_s]; idx += w1_s
    b1_s = 32; idx += b1_s  # skip b1
    w2_s = w2.size; w2.flat = bf_data[idx:idx+w2_s]; idx += w2_s
    W_eff = w2 @ w1  # 4 × 128
    return np.sum(np.abs(W_eff), axis=0)  # total importance per dim

importance = compute_importance(bf)
top16 = list(np.argsort(-importance)[:16])
bot16 = list(np.argsort(importance)[:16])
rng = np.random.RandomState(42)
rand16 = list(rng.choice(128, 16, replace=False))
# Make sure rand16 doesn't overlap too much with top16
rand16 = [d for d in rand16 if d not in top16][:12]
rand16 += list(rng.choice([d for d in range(128) if d not in top16 and d not in rand16], 4, replace=False))

print(f"Top-16 dims (highest |W_eff|): {sorted(top16)}")
print(f"Bottom-16 dims (lowest |W_eff|): {sorted(bot16)}")
print(f"Random-16 dims: {sorted(rand16)}")

# ============================================================
# Run trials for each ablation condition
# ============================================================
conditions = [
    ('intact', None, 'Intact brain'),
    ('zero_top16', top16, f'Zero top-16 dims: {sorted(top16)}'),
    ('zero_bot16', bot16, f'Zero bottom-16 dims: {sorted(bot16)}'),
    ('zero_rand16', rand16, f'Zero random-16 dims: {sorted(rand16)}'),
]

all_results = {}

for cond_name, ablate_dims, desc in conditions:
    brain = AblatableGRU()
    brain.set_params(bf)
    if ablate_dims:
        brain.set_ablation(ablate_dims)

    rng_sim = np.random.RandomState(42)
    results = []
    n_ate = 0; n_collision = 0; n_qualified = 0

    for ti in range(N_TRIALS):
        fx = rng_sim.uniform(-AX*0.7, AX*0.7)
        fy = rng_sim.uniform(AY0+4, AY1-4)
        fz = rng_sim.uniform(5, AZ-5)
        fh = rng_sim.uniform(-math.pi, math.pi)
        ht = np.zeros((1, brain.H))

        # Food (guaranteed visible-ish)
        for _ in range(30):
            a = rng_sim.uniform(-0.8, 0.8); d = rng_sim.uniform(10, 28)
            ffx = fx + d*math.sin(fh+a); ffz = fz + d*math.cos(fh+a)
            ffx = np.clip(ffx, -AX+2, AX-2); ffz = np.clip(ffz, 2, AZ-2)
            ffy = rng_sim.uniform(AY0+2, AY1-2)
            if food_visible(fx, fy, fz, fh, ffx, ffy, ffz):
                break
        food = [ffx, ffy, ffz, FOOD_R]
        qualified = food_visible(fx, fy, fz, fh, ffx, ffy, ffz)
        if qualified: n_qualified += 1

        # Obstacles
        obstacles = []
        for _ in range(N_OBS):
            for __ in range(20):
                ox = rng_sim.uniform(-AX+3, AX-3)
                oy = rng_sim.uniform(AY0+2, AY1-2)
                oz = rng_sim.uniform(3, AZ-3)
                orx = rng_sim.uniform(2, 5); ory = rng_sim.uniform(2, 5); orz = rng_sim.uniform(2, 5)
                if abs(ox-fx) < orx+FISH_BODY_R+1 and abs(oy-fy) < ory+FISH_BODY_R+1 and abs(oz-fz) < orz+FISH_BODY_R+1:
                    continue
                obstacles.append([ox, oy, oz, orx, ory, orz])
                break

        ate = False; collision = False; collision_count = 0

        for st in range(N_STEPS):
            out_of_bounds = abs(fx)>AX or fz<0 or fz>AZ or fy<AY0+0.5 or fy>AY1-0.5

            # Collision check
            for ox, oy, oz, orx, ory, orz in obstacles:
                if abs(fx-ox) < FISH_BODY_R+orx and abs(fy-oy) < FISH_BODY_R+ory and abs(fz-oz) < FISH_BODY_R+orz:
                    collision = True; collision_count += 1
                    dx=fx-ox; dy=fy-oy; dz=fz-oz
                    dist=math.hypot(dx,dy,dz)
                    if dist>0.1:
                        push=FISH_BODY_R+max(orx,ory,orz)
                        fx+=dx/dist*push; fy+=dy/dist*push; fz+=dz/dist*push
                    fy=np.clip(fy,AY0+0.1,AY1-0.1)
                    break

            dist = math.sqrt((fx-ffx)**2+(fy-ffy)**2+(fz-ffz)**2)
            just_ate = dist < FOOD_R + food[3]
            if just_ate: ate = True; break
            if out_of_bounds: break

            L, R = render_lateral(fx,fy,fz,fh,[(ffx,ffy,ffz,food[3])],obstacles)
            enc = retina1408(L,R)
            out, ht = brain.forward(enc, ht)
            lt,rt,ut,dt = float(out[0,0]),float(out[0,1]),float(out[0,2]),float(out[0,3])

            fwd=(lt+rt)/2*8; fh+=(rt-lt)*0.5
            fx+=fwd*math.sin(fh); fz+=fwd*math.cos(fh); fy+=(ut-dt)*5
            while fx>AX:fx-=2*AX
            while fx<-AX:fx+=2*AX
            while fz>AZ:fz-=AZ
            while fz<0:fz+=AZ
            fy=np.clip(fy,AY0+0.1,AY1-0.1)

        if ate: n_ate += 1
        if collision: n_collision += 1
        results.append({'ate': ate, 'collision': collision, 'collision_count': collision_count,
                        'qualified': qualified, 'steps': st+1})

    all_results[cond_name] = {
        'desc': desc,
        'n_trials': N_TRIALS,
        'n_qualified': n_qualified,
        'n_ate': n_ate,
        'n_collision': n_collision,
        'ate_rate': n_ate / max(1, n_qualified),
        'collision_rate': n_collision / N_TRIALS,
    }
    print(f"  {cond_name:15s}: ATE={n_ate}/{n_qualified}={n_ate/max(1,n_qualified):.3f}  "
          f"Collision={n_collision}/{N_TRIALS}={n_collision/N_TRIALS:.3f}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print("ABLATION RESULTS")
print(f"{'Condition':<20s} {'ATE':>8s} {'Collision':>10s} {'Desc'}")
print("-" * 70)
for cond_name in ['intact', 'zero_top16', 'zero_bot16', 'zero_rand16']:
    r = all_results[cond_name]
    print(f"  {cond_name:<18s} {r['ate_rate']:7.3f}  {r['collision_rate']:9.3f}   {r['desc']}")

print(f"\n  Top-16 dim importance: {[float(f'{importance[d]:.2f}') for d in top16]}")
print(f"  Bot-16 dim importance: {[float(f'{importance[d]:.4f}') for d in bot16]}")

csv_path = OUT_DIR / 'ablation_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['condition','n_trials','n_qualified','n_ate','ate_rate','n_collision','collision_rate'])
    for cond_name in ['intact', 'zero_top16', 'zero_bot16', 'zero_rand16']:
        r = all_results[cond_name]
        w.writerow([cond_name, r['n_trials'], r['n_qualified'], r['n_ate'],
                     round(r['ate_rate'],4), r['n_collision'], round(r['collision_rate'],4)])
print(f"Saved: {csv_path}")
print("DONE — Phase 4")

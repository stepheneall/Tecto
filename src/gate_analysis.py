"""
Phase 1: GRU Gate State Analysis
Record z, r, h, h_tilde across 4 scenarios: food-only, obs-only, mixed, empty.
50 controlled frames per scenario. Analyze gate activation patterns.

Usage: python gate_analysis.py
"""
import numpy as np, cv2, torch, json, math, csv
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# ============================================================
# GRU with internal state recording
# ============================================================
class GRUProbe:
    """GRU that returns (output, h_new, z, r, h_tilde) for circuit analysis."""
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
        """Returns (output, h_new, z_gate, r_gate, h_tilde)."""
        H = self.H
        z = 1/(1+np.exp(-np.clip(x@self.W_z.T + h@self.U_z.T + self.b_z, -10, 10)))
        r = 1/(1+np.exp(-np.clip(x@self.W_r.T + h@self.U_r.T + self.b_r, -10, 10)))
        ht_ = np.tanh(x@self.W_h.T + (r*h)@self.U_h.T + self.b_h)
        hn = (1-z)*h + z*ht_; hn *= 0.999
        out = np.tanh(np.maximum(0, hn@self.W1.T+self.b1)@self.W2.T+self.b2)
        return out, hn, z, r, ht_

# ============================================================
# Load brain
# ============================================================
BRAIN_PATH = _DATA / 'v12_mixed_H128.npy'
bf = np.load(BRAIN_PATH)
brain = GRUProbe(); brain.set_params(bf)
print(f"Brain: {BRAIN_PATH.name} ({len(bf)}p, H={brain.H})")

# ============================================================
# Generate frames for 4 scenarios
# ============================================================
N_FRAMES = 50
rng = np.random.RandomState(42)
records = []

scenarios = ['food_only', 'obs_only', 'mixed', 'empty']
print("Generating frames...")

for scenario in scenarios:
    for fi in range(N_FRAMES):
        # Fish at random position
        fx = rng.uniform(-AX*0.7, AX*0.7)
        fy = rng.uniform(AY0+4, AY1-4)
        fz = rng.uniform(5, AZ-5)
        fh = rng.uniform(-math.pi, math.pi)

        foods = []
        obstacles = []

        if scenario in ('food_only', 'mixed'):
            # Place food in front of fish
            a = rng.uniform(-0.8, 0.8); d = rng.uniform(8, 30)
            ffx = fx + d*math.sin(fh+a)
            ffz = fz + d*math.cos(fh+a)
            ffx = np.clip(ffx, -AX+2, AX-2)
            ffz = np.clip(ffz, 2, AZ-2)
            ffy = rng.uniform(AY0+2, AY1-2)
            foods = [(ffx, ffy, ffz, FOOD_R)]

        if scenario in ('obs_only', 'mixed'):
            n_obs = rng.randint(1, 4) if scenario == 'mixed' else rng.randint(1, 4)
            for _ in range(n_obs):
                for __ in range(20):
                    ox = rng.uniform(-AX+3, AX-3)
                    oy = rng.uniform(AY0+2, AY1-2)
                    oz = rng.uniform(3, AZ-3)
                    orx = rng.uniform(2, 5); ory = rng.uniform(2, 5); orz = rng.uniform(2, 5)
                    # Not inside fish
                    if abs(ox-fx) < orx+FISH_BODY_R+1 and abs(oy-fy) < ory+FISH_BODY_R+1 and abs(oz-fz) < orz+FISH_BODY_R+1:
                        continue
                    obstacles.append([ox, oy, oz, orx, ory, orz])
                    break

        # Render
        L, R = render_lateral(fx, fy, fz, fh, foods, obstacles)
        enc = retina1408(L, R)

        # Forward with initial h=0 (single-frame response, no temporal context)
        ht0 = np.zeros((1, brain.H))
        out, hn, z, r, ht_ = brain.forward(enc, ht0)

        # Compute distances
        dist_to_food = 999
        if foods:
            ffx, ffy, ffz, _ = foods[0]
            dist_to_food = math.sqrt((fx-ffx)**2 + (fy-ffy)**2 + (fz-ffz)**2)

        dist_to_obs = 999
        n_vis_obs = 0
        for ox, oy, oz, orx, ory, orz in obstacles:
            d = math.sqrt((fx-ox)**2 + (fy-oy)**2 + (fz-oz)**2)
            dist_to_obs = min(dist_to_obs, d)
            # Check if visible (rough: just check if in front hemisphere)
            rlx, rlz = rot(ox-fx, oz-fz, fh)
            if rlz > 1.0:  # in front
                n_vis_obs += 1

        lt, rt, ut, dt = float(out[0,0]), float(out[0,1]), float(out[0,2]), float(out[0,3])

        records.append({
            'scenario': scenario,
            'frame': fi,
            'fx': round(fx, 1), 'fz': round(fz, 1), 'fh': round(math.degrees(fh), 1),
            'dist_food': round(dist_to_food, 1),
            'dist_obs': round(dist_to_obs, 1),
            'n_vis_obs': n_vis_obs,
            'n_obs_total': len(obstacles),
            'z_mean': float(np.mean(z)), 'z_std': float(np.std(z)),
            'r_mean': float(np.mean(r)), 'r_std': float(np.std(r)),
            'z_min': float(np.min(z)), 'z_max': float(np.max(z)),
            'r_min': float(np.min(r)), 'r_max': float(np.max(r)),
            'h_mean': float(np.mean(hn)), 'h_std': float(np.std(hn)),
            'ht_mean': float(np.mean(ht_)), 'ht_std': float(np.std(ht_)),
            'L': lt, 'R': rt, 'U': ut, 'D': dt,
            'fwd': (lt+rt)/2,
            'turn': (rt-lt),
        })

        # Print progress
        if fi == 0:
            print(f"  {scenario}: frame 0 z_μ={np.mean(z):.4f} r_μ={np.mean(r):.4f} "
                  f"food={dist_to_food:.0f} obs={dist_to_obs:.0f}")

# ============================================================
# Analysis
# ============================================================
print(f"\nTotal records: {len(records)}")

# Group by scenario
groups = {}
for r in records:
    s = r['scenario']
    if s not in groups: groups[s] = []
    groups[s].append(r)

print(f"\n{'='*70}")
print("GATE ACTIVATION BY SCENARIO")
print(f"{'Scenario':<12s} {'z_mean':>8s} {'z_std':>8s} {'r_mean':>8s} {'r_std':>8s} {'z_range':>10s} {'r_range':>10s}")
print("-" * 70)
for s in ['food_only', 'obs_only', 'mixed', 'empty']:
    g = groups[s]
    z_m = np.mean([r['z_mean'] for r in g])
    z_s = np.mean([r['z_std'] for r in g])
    r_m = np.mean([r['r_mean'] for r in g])
    r_s = np.mean([r['r_std'] for r in g])
    z_min = np.mean([r['z_min'] for r in g])
    z_max = np.mean([r['z_max'] for r in g])
    r_min = np.mean([r['r_min'] for r in g])
    r_max = np.mean([r['r_max'] for r in g])
    print(f"  {s:<10s} {z_m:8.4f} {z_s:8.4f} {r_m:8.4f} {r_s:8.4f} "
          f"[{z_min:.3f},{z_max:.3f}] [{r_min:.3f},{r_max:.3f}]")

# Detailed: does z/r change when obstacle is close?
print(f"\n--- Gate vs obstacle distance (obs_only) ---")
obs_recs = sorted(groups['obs_only'], key=lambda r: r['dist_obs'])
for dist_bin, label in [(10, 'Near (<10)'), (20, 'Mid (10-20)'), (999, 'Far (>20)')]:
    subset = [r for r in obs_recs if r['dist_obs'] < dist_bin and (dist_bin==999 or r['dist_obs']>=dist_bin-10)]
    # Actually just do 3 bins
near = [r for r in obs_recs if r['dist_obs'] < 12]
mid = [r for r in obs_recs if 12 <= r['dist_obs'] < 25]
far = [r for r in obs_recs if r['dist_obs'] >= 25]
for label, subset in [('Near (<12)', near), ('Mid (12-25)', mid), ('Far (>25)', far)]:
    if subset:
        print(f"  {label} ({len(subset)} frames): z={np.mean([r['z_mean'] for r in subset]):.4f}  "
              f"r={np.mean([r['r_mean'] for r in subset]):.4f}  "
              f"h_std={np.mean([r['h_std'] for r in subset]):.4f}")

# Gate correlation with outputs
print(f"\n--- Gate-output correlations (across all scenarios) ---")
z_means = [r['z_mean'] for r in records]
r_means = [r['r_mean'] for r in records]
fwds = [r['fwd'] for r in records]
turns = [abs(r['turn']) for r in records]
print(f"  corr(z_mean, fwd):  {np.corrcoef(z_means, fwds)[0,1]:+.4f}")
print(f"  corr(z_mean, |turn|): {np.corrcoef(z_means, turns)[0,1]:+.4f}")
print(f"  corr(r_mean, fwd):  {np.corrcoef(r_means, fwds)[0,1]:+.4f}")
print(f"  corr(r_mean, |turn|): {np.corrcoef(r_means, turns)[0,1]:+.4f}")

# ============================================================
# Save detailed frame records
# ============================================================
csv_path = OUT_DIR / 'gate_records.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader()
    for r in records:
        w.writerow(r)
print(f"\nSaved: {csv_path}")

# ============================================================
# Plot: gate mean per scenario
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. z distribution per scenario
ax = axes[0, 0]
scenario_labels = ['food_only', 'obs_only', 'mixed', 'empty']
colors = ['#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
for s, c in zip(scenario_labels, colors):
    z_vals = [r['z_mean'] for r in groups[s]]
    ax.hist(z_vals, bins=20, alpha=0.5, color=c, label=s)
ax.set_xlabel('mean z (update gate)'); ax.set_ylabel('count')
ax.set_title('Update Gate (z) Distribution by Scenario')
ax.legend(fontsize=8)

# 2. r distribution per scenario
ax = axes[0, 1]
for s, c in zip(scenario_labels, colors):
    r_vals = [r['r_mean'] for r in groups[s]]
    ax.hist(r_vals, bins=20, alpha=0.5, color=c, label=s)
ax.set_xlabel('mean r (reset gate)'); ax.set_ylabel('count')
ax.set_title('Reset Gate (r) Distribution by Scenario')
ax.legend(fontsize=8)

# 3. z vs r scatter colored by scenario
ax = axes[0, 2]
for s, c in zip(scenario_labels, colors):
    g = groups[s]
    ax.scatter([r['z_mean'] for r in g], [r['r_mean'] for r in g],
               c=c, label=s, alpha=0.6, s=20)
ax.set_xlabel('z mean'); ax.set_ylabel('r mean')
ax.set_title('z vs r by Scenario')
ax.legend(fontsize=8)

# 4. z vs obstacle distance (obs_only)
ax = axes[1, 0]
g = groups['obs_only']
ax.scatter([r['dist_obs'] for r in g], [r['z_mean'] for r in g],
           c='#d62728', alpha=0.7, s=20)
ax.set_xlabel('distance to nearest obstacle'); ax.set_ylabel('z mean')
ax.set_title('Update Gate vs Obstacle Distance (obs_only)')

# 5. r vs obstacle distance
ax = axes[1, 1]
ax.scatter([r['dist_obs'] for r in g], [r['r_mean'] for r in g],
           c='#d62728', alpha=0.7, s=20)
ax.set_xlabel('distance to nearest obstacle'); ax.set_ylabel('r mean')
ax.set_title('Reset Gate vs Obstacle Distance (obs_only)')

# 6. Gate mean bar chart
ax = axes[1, 2]
x = np.arange(4); w = 0.35
z_means_by_s = [np.mean([r['z_mean'] for r in groups[s]]) for s in scenario_labels]
r_means_by_s = [np.mean([r['r_mean'] for r in groups[s]]) for s in scenario_labels]
z_errs = [np.std([r['z_mean'] for r in groups[s]]) for s in scenario_labels]
r_errs = [np.std([r['r_mean'] for r in groups[s]]) for s in scenario_labels]
ax.bar(x - w/2, z_means_by_s, w, yerr=z_errs, color='#1f77b4', label='z (update)')
ax.bar(x + w/2, r_means_by_s, w, yerr=r_errs, color='#ff7f0e', label='r (reset)')
ax.set_xticks(x); ax.set_xticklabels(scenario_labels, fontsize=8)
ax.set_ylabel('gate mean'); ax.set_title('Gate Activation by Scenario')
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = OUT_DIR / 'gate_analysis.png'
fig.savefig(fig_path, dpi=120)
print(f"Saved: {fig_path}")
print("DONE — Phase 1")

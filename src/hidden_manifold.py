"""
Phase 2: Hidden State Manifold Analysis
Collect hidden states across multi-step trajectories in different scenarios.
PCA projection to see if h occupies different subspaces for food vs obstacle.

Usage: python hidden_manifold.py
"""
import numpy as np, cv2, torch, json, math, csv
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
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
N_TRAJECTORIES = 15; N_STEPS = 60

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
BRAIN_PATH = _DATA / 'v12_mixed_H128.npy'
bf = np.load(BRAIN_PATH)
brain = GRU(); brain.set_params(bf)
print(f"Brain: {BRAIN_PATH.name} ({len(bf)}p, H={brain.H})")

rng = np.random.RandomState(42)
all_h = []   # list of (h_vector, scenario, dist_food, dist_obs, turn, fwd)
all_labels = []

print("Running trajectories...")
for scenario in ['food_only', 'obs_only', 'mixed']:
    for ti in range(N_TRAJECTORIES):
        fx = rng.uniform(-AX*0.7, AX*0.7)
        fy = rng.uniform(AY0+4, AY1-4)
        fz = rng.uniform(5, AZ-5)
        fh = rng.uniform(-math.pi, math.pi)
        ht = np.zeros((1, brain.H))

        # Place food
        food = None
        if scenario in ('food_only', 'mixed'):
            a = rng.uniform(-0.8, 0.8); d = rng.uniform(8, 30)
            ffx = fx + d*math.sin(fh+a); ffz = fz + d*math.cos(fh+a)
            ffx = np.clip(ffx, -AX+2, AX-2); ffz = np.clip(ffz, 2, AZ-2)
            ffy = rng.uniform(AY0+2, AY1-2)
            food = [ffx, ffy, ffz, FOOD_R]

        # Place obstacles (fixed for this trajectory)
        obstacles = []
        if scenario in ('obs_only', 'mixed'):
            n_obs = rng.randint(1, 4)
            for _ in range(n_obs):
                for __ in range(20):
                    ox = rng.uniform(-AX+3, AX-3); oy = rng.uniform(AY0+2, AY1-2)
                    oz = rng.uniform(3, AZ-3)
                    orx = rng.uniform(2, 5); ory = rng.uniform(2, 5); orz = rng.uniform(2, 5)
                    if abs(ox-fx) < orx+FISH_BODY_R+1 and abs(oy-fy) < ory+FISH_BODY_R+1 and abs(oz-fz) < orz+FISH_BODY_R+1:
                        continue
                    obstacles.append([ox, oy, oz, orx, ory, orz])
                    break

        traj_h = []
        for st in range(N_STEPS):
            foods_list = [food] if food else []
            L, R = render_lateral(fx, fy, fz, fh, foods_list, obstacles)
            enc = retina1408(L, R)
            out, ht = brain.forward(enc, ht)

            lt, rt, ut, dt = float(out[0,0]), float(out[0,1]), float(out[0,2]), float(out[0,3])
            fwd = (lt+rt)/2 * 8
            turn = (rt-lt) * 0.5
            fh += turn
            fx += fwd*math.sin(fh); fz += fwd*math.cos(fh); fy += (ut-dt)*5

            # Distances
            df = math.sqrt((fx-food[0])**2+(fy-food[1])**2+(fz-food[2])**2) if food else 999
            do = 999
            for ox, oy, oz, orx, ory, orz in obstacles:
                do = min(do, math.sqrt((fx-ox)**2+(fy-oy)**2+(fz-oz)**2))

            # Store
            all_h.append(ht[0].copy())
            all_labels.append({
                'scenario': scenario, 'ti': ti, 'step': st,
                'dist_food': round(df, 1), 'dist_obs': round(do, 1),
                'fwd': round(fwd, 2), 'turn': round(turn, 4),
                'L': round(lt, 3), 'R': round(rt, 3),
            })

            # Respawn if too far off
            if abs(fx) > AX or fz < 0 or fz > AZ or fy < AY0+0.5 or fy > AY1-0.5:
                fx = rng.uniform(-AX*0.7, AX*0.7); fy = rng.uniform(AY0+4, AY1-4)
                fz = rng.uniform(5, AZ-5); fh = rng.uniform(-math.pi, math.pi)
                ht = np.zeros((1, brain.H))

    n = len(all_h)
    print(f"  {scenario}: {N_TRAJECTORIES} traj, {len([l for l in all_labels if l['scenario']==scenario])} frames")

# ============================================================
# PCA
# ============================================================
print(f"\nTotal hidden state vectors: {len(all_h)}")
H_matrix = np.array(all_h)
print(f"H matrix shape: {H_matrix.shape}")

pca = PCA(n_components=3)
H_pca = pca.fit_transform(H_matrix)
print(f"PCA explained variance: {pca.explained_variance_ratio_}")

# ============================================================
# Categorize frames by behavioral context
# ============================================================
# "approaching food": dist_food decreasing, food visible
# "avoiding obstacle": dist_obs < 15, turn magnitude high
# "cruising": no food/obs nearby
def categorize(label):
    if label['dist_food'] < 20 and label['scenario'] in ('food_only', 'mixed'):
        return 'approach_food'
    elif label['dist_obs'] < 15 and abs(label['turn']) > 0.2:
        return 'avoid_obs'
    elif label['dist_obs'] < 15:
        return 'near_obs'
    elif label['dist_food'] < 999 and label['dist_food'] >= 20:
        return 'seek_food'
    else:
        return 'cruise'

categories = [categorize(l) for l in all_labels]
cat_counts = {}
for c in categories:
    cat_counts[c] = cat_counts.get(c, 0) + 1
print(f"Categories: {cat_counts}")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

cat_colors = {
    'approach_food': '#2ca02c',
    'seek_food': '#98df8a',
    'avoid_obs': '#d62728',
    'near_obs': '#ff9896',
    'cruise': '#7f7f7f',
}

cats_sorted = sorted(set(categories))

# 1. PC1 vs PC2 colored by scenario
ax = axes[0, 0]
scenario_colors = {'food_only': '#2ca02c', 'obs_only': '#d62728', 'mixed': '#9467bd'}
for s in ['food_only', 'obs_only', 'mixed']:
    mask = [l['scenario']==s for l in all_labels]
    ax.scatter(H_pca[mask, 0], H_pca[mask, 1], c=scenario_colors[s], label=s,
               alpha=0.4, s=8)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Hidden States by Scenario')
ax.legend(fontsize=8)

# 2. PC1 vs PC2 colored by behavioral category
ax = axes[0, 1]
for c in cats_sorted:
    mask = [cat==c for cat in categories]
    if sum(mask) > 5:
        ax.scatter(H_pca[mask, 0], H_pca[mask, 1], c=cat_colors[c], label=c,
                   alpha=0.4, s=8)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Hidden States by Behavioral Context')
ax.legend(fontsize=7)

# 3. PC1 vs PC3
ax = axes[1, 0]
for c in cats_sorted:
    mask = [cat==c for cat in categories]
    if sum(mask) > 5:
        ax.scatter(H_pca[mask, 0], H_pca[mask, 2], c=cat_colors[c], label=c,
                   alpha=0.4, s=8)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
ax.set_title('PC1 vs PC3 by Context')
ax.legend(fontsize=7)

# 4. PC space trajectory overlay for a few representative traj
ax = axes[1, 1]
# Pick one trajectory from each scenario
for s, color in [('food_only', '#2ca02c'), ('obs_only', '#d62728'), ('mixed', '#9467bd')]:
    for ti in [0, 5, 10]:
        mask = [i for i, l in enumerate(all_labels) if l['scenario']==s and l['ti']==ti]
        if mask:
            points = H_pca[mask]
            ax.plot(points[:, 0], points[:, 1], '-', color=color, alpha=0.5, lw=0.8)
            ax.scatter(points[0, 0], points[0, 1], marker='o', color=color, s=15, alpha=0.8)
            ax.scatter(points[-1, 0], points[-1, 1], marker='s', color=color, s=15, alpha=0.8)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Trajectory Paths in PC Space (o=start, ■=end)')

plt.tight_layout()
fig_path = OUT_DIR / 'hidden_manifold.png'
fig.savefig(fig_path, dpi=120)
print(f"\nSaved: {fig_path}")

# ============================================================
# Cross-scenario hidden state similarity
# ============================================================
print(f"\n--- Cross-scenario hidden state similarity ---")
scenario_h = {}
for s in ['food_only', 'obs_only', 'mixed']:
    mask = [i for i, l in enumerate(all_labels) if l['scenario']==s]
    scenario_h[s] = H_matrix[mask]

for s1 in ['food_only', 'obs_only', 'mixed']:
    for s2 in ['food_only', 'obs_only', 'mixed']:
        # Mean cosine similarity between scenarios
        mean_h1 = scenario_h[s1].mean(axis=0)
        mean_h2 = scenario_h[s2].mean(axis=0)
        cos_sim = np.dot(mean_h1, mean_h2) / (np.linalg.norm(mean_h1)*np.linalg.norm(mean_h2))
        print(f"  cos_sim({s1}, {s2}) = {cos_sim:.4f}")

# Save hidden states CSV (too big for raw, save PCA + labels)
csv_path = OUT_DIR / 'hidden_pca.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['PC1','PC2','PC3','scenario','ti','step','dist_food','dist_obs','fwd','turn','category'])
    for i, l in enumerate(all_labels):
        w.writerow([H_pca[i,0], H_pca[i,1], H_pca[i,2],
                     l['scenario'], l['ti'], l['step'],
                     l['dist_food'], l['dist_obs'], l['fwd'], l['turn'],
                     categories[i]])
print(f"Saved: {csv_path}")
print("DONE — Phase 2")

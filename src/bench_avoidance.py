"""
Obstacle avoidance benchmark — 500 fish, 15 obstacles, NO food, pure torus.
Measures: collision rate, steps-before-collision, survival rate.
V12 brain. Pure avoidance — no food motivation, just obstacle navigation.

Usage: python bench_avoidance.py [brain.npy] [n_fish] [n_steps]
"""
import numpy as np, cv2, torch, json, math, time, sys, csv
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
FISH_BODY_R = 1.5  # same as obs_test collision radius
N_OBS = 15

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

def object_visible(fx, fy, fz, fh, ox, oy, oz, osize):
    """Check if ANY part of object is visible in either eye."""
    L_ang = fh - EYE_ANGLE; L_ex = fx - EYE_OFFSET*math.cos(fh); L_ez = fz + EYE_OFFSET*math.sin(fh)
    R_ang = fh + EYE_ANGLE; R_ex = fx + EYE_OFFSET*math.cos(fh); R_ez = fz - EYE_OFFSET*math.sin(fh)
    FL = 80; SZ_W = 280; SZ_H = 280; cx = SZ_W//2; cy = SZ_H//2
    for ex, ez, eh in [(L_ex, L_ez, L_ang), (R_ex, R_ez, R_ang)]:
        rlx, rlz = rot(ox-ex, oz-ez, eh); rly = oy - fy
        if rlz > 0.3:
            px = int(cx + FL*rlx/max(rlz, 0.5))
            py = int(cy + FL*rly/max(rlz, 0.5))
            pr = max(2, int(FL*osize/max(rlz, 0.5)))
            if 0 <= px < SZ_W and 0 < py < SZ_H and pr >= 2:
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
print(f"N={n_fish} fish, {n_steps} max steps, {N_OBS} obstacles, NO food")
print()

rng = np.random.RandomState(42)
results = []

for fi in range(n_fish):
    # ---- spawn fish (away from arena edges) ----
    fx = rng.uniform(-AX*0.8, AX*0.8)
    fy = rng.uniform(AY0+3, AY1-3)
    fz = rng.uniform(5, AZ-5)
    fh = rng.uniform(-math.pi, math.pi)
    ht = np.zeros((1, brain.H))

    # ---- spawn obstacles ----
    obstacles = []
    for _ in range(N_OBS):
        for __ in range(30):
            ox = rng.uniform(-AX+4, AX-4)
            oy = rng.uniform(AY0+3, AY1-3)
            oz = rng.uniform(4, AZ-4)
            orx = rng.uniform(2, 6)
            ory = rng.uniform(2, 6)
            orz = rng.uniform(2, 6)
            # Don't spawn inside fish
            if abs(ox-fx) < orx+FISH_BODY_R+2 and abs(oy-fy) < ory+FISH_BODY_R+2 and abs(oz-fz) < orz+FISH_BODY_R+2:
                continue
            # Don't overlap other obstacles too much
            ok = True
            for oox, ooy, ooz, oorx, oory, oorz in obstacles:
                if abs(ox-oox) < orx+oorx+1 and abs(oy-ooy) < ory+oory+1 and abs(oz-ooz) < orz+oorz+1:
                    ok = False; break
            if ok:
                obstacles.append([ox, oy, oz, orx, ory, orz])
                break

    # Check if ANY obstacle is visible at spawn (qualified)
    any_visible = False
    closest_obs_dist = 999
    for ox, oy, oz, orx, ory, orz in obstacles:
        osize = max(orx, ory, orz)
        if object_visible(fx, fy, fz, fh, ox, oy, oz, osize):
            any_visible = True
        d = math.sqrt((fx-ox)**2 + (fy-oy)**2 + (fz-oz)**2)
        closest_obs_dist = min(closest_obs_dist, d)

    n_visible = sum(1 for ox, oy, oz, orx, ory, orz in obstacles
                    if object_visible(fx, fy, fz, fh, ox, oy, oz, max(orx, ory, orz)))

    # ---- run ----
    collision = False; collision_step = 0; total_collisions = 0
    steps_survived = 0; min_obs_dist = closest_obs_dist

    for st in range(n_steps):
        out_of_bounds = abs(fx) > AX or fz < 0 or fz > AZ or fy < AY0+0.5 or fy > AY1-0.5

        # ---- check obstacle collisions (same as obs_test) ----
        hit_this_step = False
        for ox, oy, oz, orx, ory, orz in obstacles:
            if abs(fx-ox) < FISH_BODY_R+orx and abs(fy-oy) < FISH_BODY_R+ory and abs(fz-oz) < FISH_BODY_R+orz:
                hit_this_step = True
                total_collisions += 1
                if not collision:
                    collision = True
                    collision_step = st + 1
                # Push fish away (same as obs_test)
                dx = fx - ox; dy = fy - oy; dz = fz - oz
                dist = math.hypot(dx, dy, dz)
                if dist > 0.1:
                    push = FISH_BODY_R + max(orx, ory, orz)
                    fx += dx/dist * push
                    fy += dy/dist * push
                    fz += dz/dist * push
                fy = np.clip(fy, AY0+0.1, AY1-0.1)
                break  # only count one collision per step

        # Update closest obs distance
        for ox, oy, oz, orx, ory, orz in obstacles:
            d = math.sqrt((fx-ox)**2 + (fy-oy)**2 + (fz-oz)**2)
            min_obs_dist = min(min_obs_dist, d)

        if out_of_bounds:
            break

        # ---- vision + brain ----
        L, R = render_lateral(fx, fy, fz, fh, [], obstacles)
        enc = retina1408(L, R)
        out, ht = brain.forward(enc, ht)
        lt, rt, ut, dt = float(out[0,0]), float(out[0,1]), float(out[0,2]), float(out[0,3])

        # ---- move fish ----
        fwd = (lt+rt)/2 * 8
        fh += (rt-lt) * 0.5
        fx = fx + fwd*math.sin(fh)
        fz = fz + fwd*math.cos(fh)
        fy = fy + (ut-dt) * 5

        # Torus wrap
        while fx > AX: fx -= 2*AX
        while fx < -AX: fx += 2*AX
        while fz > AZ: fz -= AZ
        while fz < 0: fz += AZ
        fy = np.clip(fy, AY0+0.1, AY1-0.1)

        steps_survived = st + 1

    results.append({
        'idx': fi,
        'collision': collision,
        'collision_step': collision_step,
        'total_collisions': total_collisions,
        'steps': steps_survived,
        'any_visible': any_visible,
        'n_visible': n_visible,
        'spawn_obs_dist': round(closest_obs_dist, 1),
        'min_obs_dist': round(min_obs_dist, 1),
        'survived': not collision and not out_of_bounds and steps_survived >= n_steps,
    })

    if (fi+1) % 50 == 0 or fi == 0:
        c_so_far = sum(1 for r in results if r['collision'])
        surv_so_far = sum(1 for r in results if r['survived'])
        print(f"  [{fi+1:>4d}/{n_fish}] Collisions:{c_so_far}/{fi+1}={c_so_far/(fi+1):.2f}  "
              f"Survived:{surv_so_far}/{fi+1}={surv_so_far/(fi+1):.2f}")

# ============================================================
# Results
# ============================================================
collided = [r for r in results if r['collision']]
survived = [r for r in results if r['survived']]
visible = [r for r in results if r['any_visible']]
blind = [r for r in results if not r['any_visible']]

print(f"\n{'='*70}")
print(f"BENCHMARK: Obstacle Avoidance (no food, torus)")
print(f"  Brain: {brain_path.name}  N={n_fish}  MaxSteps={n_steps}  Obstacles={N_OBS}")
print(f"{'='*70}")
print(f"  Total fish:          {n_fish}")
print(f"  Collided (≥1 hit):   {len(collided)} ({len(collided)/n_fish*100:.1f}%)")
print(f"  Collision-free:      {n_fish - len(collided)} ({(n_fish - len(collided))/n_fish*100:.1f}%)")
print(f"  Full survival (no collision, no OOB):  {len(survived)} ({len(survived)/n_fish*100:.1f}%)")
print()

if collided:
    print(f"  --- Collision fish ---")
    print(f"  Mean collisions/fish:     {np.mean([r['total_collisions'] for r in collided]):.2f}")
    print(f"  Mean step of first hit:   {np.mean([r['collision_step'] for r in collided]):.1f}")
    print(f"  Median step of first hit: {np.median([r['collision_step'] for r in collided]):.1f}")

print(f"\n  --- Visible vs blind at spawn ---")
print(f"  Saw obstacle at spawn:  {len(visible)} fish")
if visible:
    v_coll = sum(1 for r in visible if r['collision'])
    print(f"    → Collided: {v_coll}/{len(visible)} = {v_coll/len(visible):.3f}")
    print(f"    → Mean n_visible: {np.mean([r['n_visible'] for r in visible]):.1f}")
print(f"  Saw NO obstacle:        {len(blind)} fish")
if blind:
    b_coll = sum(1 for r in blind if r['collision'])
    print(f"    → Collided: {b_coll}/{len(blind)} = {b_coll/len(blind):.3f}")

print(f"\n  --- By number of visible obstacles at spawn ---")
for nv in sorted(set(r['n_visible'] for r in results)):
    subset = [r for r in results if r['n_visible'] == nv]
    c_sub = sum(1 for r in subset if r['collision'])
    print(f"  Vis={nv}: {len(subset):>4d} fish,  collision rate={c_sub/len(subset):.3f}")

# Save CSV
csv_path = OUT_DIR / f'avoidance_bench_{brain_path.stem}.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['idx','collision','collision_step','total_collisions',
                                       'steps','any_visible','n_visible','spawn_obs_dist',
                                       'min_obs_dist','survived'])
    w.writeheader()
    for r in results:
        w.writerow(r)
print(f"\nSaved: {csv_path}")
print("DONE")

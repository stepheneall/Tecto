"""
Realistic mixed benchmark — food + sparse obstacles (2-3).
This matches training distribution: food always present, obstacles occasional.
Measures: ATE rate, collision rate, collision-free survival.
300 fish, V12 brain.

Usage: python bench_mixed.py [brain.npy]
"""
import numpy as np, cv2, torch, json, math, sys, csv
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
N_STEPS = 400; N_FISH = 300; N_OBS = 3

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
brain_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    _DATA / 'v12_mixed_H128.npy'
bf = np.load(brain_path)
brain = GRU(); brain.set_params(bf)
print(f"Brain: {brain_path.name} ({len(bf)}p, H={brain.H})")
print(f"N={N_FISH} fish, {N_STEPS} max steps, {N_OBS} obstacles + 1 food")
print()

rng = np.random.RandomState(42)
results = []

for fi in range(N_FISH):
    # Spawn fish
    fx = rng.uniform(-AX*0.8, AX*0.8)
    fy = rng.uniform(AY0+3, AY1-3)
    fz = rng.uniform(5, AZ-5)
    fh = rng.uniform(-math.pi, math.pi)
    ht = np.zeros((1, brain.H))

    # Spawn food (static, random position, always in front-ish)
    for _ in range(30):
        a = rng.uniform(-1.0, 1.0); d = rng.uniform(10, 30)
        ffx = fx + d*math.sin(fh+a); ffz = fz + d*math.cos(fh+a)
        ffy = rng.uniform(AY0+2, AY1-2)
        ffx = np.clip(ffx, -AX+2, AX-2)
        ffz = np.clip(ffz, 2, AZ-2)
        if food_visible(fx, fy, fz, fh, ffx, ffy, ffz):
            break
    food = [ffx, ffy, ffz, rng.uniform(3, 7)]

    # Spawn obstacles (not on food, not on fish)
    obstacles = []
    for _ in range(N_OBS):
        for __ in range(30):
            ox = rng.uniform(-AX+4, AX-4)
            oy = rng.uniform(AY0+3, AY1-3)
            oz = rng.uniform(4, AZ-4)
            orx = rng.uniform(2, 5); ory = rng.uniform(2, 5); orz = rng.uniform(2, 5)
            # Not inside fish
            if abs(ox-fx) < orx+FISH_BODY_R+3 and abs(oy-fy) < ory+FISH_BODY_R+3 and abs(oz-fz) < orz+FISH_BODY_R+3:
                continue
            # Not inside food
            if abs(ox-food[0]) < orx+food[3]+2 and abs(oy-food[1]) < ory+food[3]+2 and abs(oz-food[2]) < orz+food[3]+2:
                continue
            obstacles.append([ox, oy, oz, orx, ory, orz])
            break

    qualified = food_visible(fx, fy, fz, fh, ffx, ffy, ffz)
    ate = False; collision = False; collision_count = 0
    first_collision_step = 0
    start_dist = math.sqrt((fx-ffx)**2 + (fy-ffy)**2 + (fz-ffz)**2)

    for st in range(N_STEPS):
        out_of_bounds = abs(fx) > AX or fz < 0 or fz > AZ or fy < AY0+0.5 or fy > AY1-0.5

        # Collision check
        for ox, oy, oz, orx, ory, orz in obstacles:
            if abs(fx-ox) < FISH_BODY_R+orx and abs(fy-oy) < FISH_BODY_R+ory and abs(fz-oz) < FISH_BODY_R+orz:
                collision_count += 1
                if not collision:
                    collision = True
                    first_collision_step = st + 1
                # Push away
                dx = fx - ox; dy = fy - oy; dz = fz - oz
                dist = math.hypot(dx, dy, dz)
                if dist > 0.1:
                    push = FISH_BODY_R + max(orx, ory, orz)
                    fx += dx/dist * push
                    fy += dy/dist * push
                    fz += dz/dist * push
                fy = np.clip(fy, AY0+0.1, AY1-0.1)
                break

        # Eat check
        dist = math.sqrt((fx-ffx)**2 + (fy-ffy)**2 + (fz-ffz)**2)
        just_ate = dist < FOOD_R + food[3]

        if just_ate:
            ate = True
            break
        if out_of_bounds:
            break

        # Render + brain
        L, R = render_lateral(fx, fy, fz, fh, [(ffx, ffy, ffz, food[3])], obstacles)
        enc = retina1408(L, R)
        out, ht = brain.forward(enc, ht)
        lt, rt, ut, dt = float(out[0,0]), float(out[0,1]), float(out[0,2]), float(out[0,3])

        fwd = (lt+rt)/2 * 8
        fh += (rt-lt) * 0.5
        fx = fx + fwd*math.sin(fh)
        fz = fz + fwd*math.cos(fh)
        fy = fy + (ut-dt) * 5

        while fx > AX: fx -= 2*AX
        while fx < -AX: fx += 2*AX
        while fz > AZ: fz -= AZ
        while fz < 0: fz += AZ
        fy = np.clip(fy, AY0+0.1, AY1-0.1)

    results.append({
        'idx': fi,
        'qualified': qualified,
        'ate': ate,
        'collision': collision,
        'collision_count': collision_count,
        'first_collision_step': first_collision_step,
        'steps': st + 1,
        'start_dist': round(start_dist, 1),
        'end_dist': round(dist, 1),
    })

    if (fi+1) % 50 == 0 or fi == 0:
        q = [r for r in results if r['qualified']]
        qa = sum(1 for r in q if r['ate'])
        col = sum(1 for r in results if r['collision'])
        print(f"  [{fi+1:>3d}/{N_FISH}] "
              f"Q_ATE:{qa}/{len(q)}={qa/max(1,len(q)):.2f}  "
              f"Collision:{col}/{fi+1}={col/(fi+1):.2f}")

# ============================================================
q_results = [r for r in results if r['qualified']]
q_ate = sum(1 for r in q_results if r['ate'])
all_ate = sum(1 for r in results if r['ate'])
collided = [r for r in results if r['collision']]
no_collision_ate = sum(1 for r in results if not r['collision'] and r['ate'])

print(f"\n{'='*70}")
print(f"BENCHMARK: Realistic Mixed (food + {N_OBS} obstacles)")
print(f"  Brain: {brain_path.name}  N={N_FISH}  MaxSteps={N_STEPS}")
print(f"{'='*70}")
print(f"  Qualified (saw food):  {len(q_results)}/{N_FISH}")
print(f"  Qualified ATE rate:    {q_ate}/{len(q_results)} = {q_ate/max(1,len(q_results)):.3f}")
print(f"  All-fish ATE rate:     {all_ate}/{N_FISH} = {all_ate/N_FISH:.3f}")
print()
print(f"  Fish that collided:    {len(collided)}/{N_FISH} = {len(collided)/N_FISH:.3f}")
print(f"  Collision-free:        {N_FISH - len(collided)}/{N_FISH} = {(N_FISH-len(collided))/N_FISH:.3f}")
if collided:
    print(f"  Mean collisions/collided fish: {np.mean([r['collision_count'] for r in collided]):.2f}")
    print(f"  Mean step of first hit:        {np.mean([r['first_collision_step'] for r in collided]):.1f}")
    print(f"  Median step of first hit:      {np.median([r['first_collision_step'] for r in collided]):.1f}")
print()
print(f"  --- Cross-tabulation ---")
cc_ate = sum(1 for r in results if r['collision'] and r['ate'])
cc_noate = sum(1 for r in results if r['collision'] and not r['ate'])
nc_ate = sum(1 for r in results if not r['collision'] and r['ate'])
nc_noate = sum(1 for r in results if not r['collision'] and not r['ate'])
print(f"  Collision + ATE:    {cc_ate}")
print(f"  Collision + No ATE: {cc_noate}")
print(f"  No Collision + ATE: {nc_ate}")
print(f"  No Collision + No ATE: {nc_noate}")
print(f"  → ATE rate given no-collision: {nc_ate/max(1,nc_ate+nc_noate):.3f}")
print(f"  → ATE rate given collision:    {cc_ate/max(1,cc_ate+cc_noate):.3f}")

csv_path = OUT_DIR / f'mixed_bench_{brain_path.stem}.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['idx','qualified','ate','collision','collision_count',
                                       'first_collision_step','steps','start_dist','end_dist'])
    w.writeheader()
    for r in results:
        w.writerow(r)
print(f"\nSaved: {csv_path}")
print("DONE")

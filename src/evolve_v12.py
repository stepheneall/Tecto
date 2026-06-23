"""
V12 evolution fine-tuning — natural selection on top of pretrained brain.
Pretrained GRU(128, 1408→4) seeds the population; random-init baseline as control.
Arena: 15 food, 8 obstacles, energy-based selection, torus wrap.

Usage: python evolve_v12.py
"""
import numpy as np, cv2, torch, json, math, time, csv
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from collections import defaultdict

# Paths resolved relative to this file's location (portable)
SRC_DIR = Path(__file__).parent
_ROOT = SRC_DIR.parent
_DATA = _ROOT / 'data'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIM = 384; POOL_H, POOL_W = 4, 8; N_SPATIAL = 32; MICRO_H2 = 8; MICRO_OUT = 256; H = 128
GRU_IN = 384+384+384+256; GRU_OUT = 4
AX = 36; AY0, AY1 = 0, 30; AZ = 60
EYE_OFFSET = 3.0; EYE_ANGLE = math.radians(25.0)
FOOD_R = 5.0; FISH_R = 1.5
N_GRU_PARAMS = 3*(H*GRU_IN + H*H + H) + 32*H + 32 + 4*32 + 4  # ~594K
OUT_DIR = _ROOT / 'evo_finetune'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def ts(): return time.strftime('%H:%M:%S')
def log(msg): print(f"[{ts()}] {msg}", flush=True)

# ============================================================
log("Loading DINOv2 + MicroNet...")
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

# ============================================================
# V12-style rendering
# ============================================================
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
# GRU — V12 architecture (1408→128→4), vectorized for population
# ============================================================
def gru_forward_vectorized(params_batch, x, h):
    """params_batch: (P, N_PARAMS), x: (P,1408), h: (P,128) -> out (P,4), h_new (P,128)
    Each fish i uses params[i] and input[i]."""
    P = params_batch.shape[0]; D = H; off = 0
    def g(s1, s2=None):
        nonlocal off
        n = s1 if s2 is None else s1*s2
        v = params_batch[:, off:off+n]
        off += n
        return v.reshape(P, s1, s2) if s2 is not None else v.reshape(P, s1)
    Wz = g(D, GRU_IN); Uz = g(D, D); bz = g(D)
    Wr = g(D, GRU_IN); Ur = g(D, D); br = g(D)
    Wh = g(D, GRU_IN); Uh = g(D, D); bh = g(D)
    W1 = g(32, D); b1 = g(32); W2 = g(4, 32); b2 = g(4)

    # 'ni,nji->nj': per-fish pairing (x[i] with W[i])
    z = 1/(1+np.exp(-np.clip(np.einsum('ni,nji->nj', x, Wz) + np.einsum('ni,nji->nj', h, Uz) + bz, -10, 10)))
    r = 1/(1+np.exp(-np.clip(np.einsum('ni,nji->nj', x, Wr) + np.einsum('ni,nji->nj', h, Ur) + br, -10, 10)))
    ht_ = np.tanh(np.einsum('ni,nji->nj', x, Wh) + np.einsum('ni,nji->nj', r*h, Uh) + bh)
    hn = (1-z)*h + z*ht_; hn *= 0.999
    mid = np.maximum(0, np.einsum('ni,nji->nj', hn, W1) + b1)
    out = np.tanh(np.einsum('ni,nji->nj', mid, W2) + b2)
    return out, hn

# ============================================================
# Load V12 pretrained brain
# ============================================================
BRAIN_PATH = _DATA / 'v12_mixed_H128.npy'
v12_bf = np.load(BRAIN_PATH)
assert len(v12_bf) == N_GRU_PARAMS, f"Param mismatch: {len(v12_bf)} vs expected {N_GRU_PARAMS}"
log(f"Loaded V12 brain: {len(v12_bf):,} params")

# ============================================================
# Arena — torus wrap, energy metabolism
# ============================================================
class Arena:
    def __init__(self, n, rng, n_food=15, n_obs=8):
        self.n = n; self.rng = rng
        self.fx = np.zeros(n); self.fy = np.zeros(n); self.fz = np.zeros(n); self.fh = np.zeros(n)
        self.energy = np.zeros(n); self.alive = np.ones(n, bool)
        self.food_eaten = np.zeros(n); self.collisions = np.zeros(n)
        self.obs = []; self.foods = []; self._gen_obs(n_obs); self._sf(n_food)

    def _gen_obs(self, n_obs):
        for _ in range(n_obs):
            ox = self.rng.uniform(-AX+4, AX-4); oy = self.rng.uniform(AY0+3, AY1-3)
            oz = self.rng.uniform(4, AZ-4)
            self.obs.append([ox, oy, oz,
                            self.rng.uniform(2, 6), self.rng.uniform(2, 6), self.rng.uniform(2, 6)])

    def _sf(self, n):
        for _ in range(n):
            for _ in range(50):
                fx = self.rng.uniform(-AX+5, AX-5)
                fy = self.rng.uniform(AY0+3, AY1-3)
                fz = self.rng.uniform(10, AZ-10)
                fr = self.rng.uniform(3, 8)
                ok = True
                for ox, oy, oz, orx, ory, orz in self.obs:
                    if abs(fx-ox) < orx+fr+2 and abs(fy-oy) < ory+fr+2 and abs(fz-oz) < orz+fr+2:
                        ok = False; break
                if ok:
                    self.foods.append([fx, fy, fz, fr])
                    break

    def reset(self):
        self.fx[:] = self.rng.uniform(-AX*0.7, AX*0.7, self.n)
        self.fy[:] = self.rng.uniform(AY0+4, AY1-4, self.n)
        self.fz[:] = self.rng.uniform(5, AZ-5, self.n)
        self.fh[:] = self.rng.uniform(-math.pi, math.pi, self.n)
        self.energy[:] = 150; self.alive[:] = True
        self.food_eaten[:] = 0; self.collisions[:] = 0
        self.foods = []; self._sf(15)

    def step(self, lt, rt, ut, dt):
        lt = np.asarray(lt); rt = np.asarray(rt); ut = np.asarray(ut); dt = np.asarray(dt)
        fwd = (lt+rt)/2 * 8; turn = (rt-lt) * 0.5
        self.fh += turn
        nx = self.fx + fwd*np.sin(self.fh)
        nz = self.fz + fwd*np.cos(self.fh)
        ny = self.fy + (ut-dt) * 5

        # Torus wrap
        alive_idx = np.where(self.alive)[0]
        for i in alive_idx:
            while nx[i] > AX: nx[i] -= 2*AX
            while nx[i] < -AX: nx[i] += 2*AX
            while nz[i] > AZ: nz[i] -= AZ
            while nz[i] < 0: nz[i] += AZ
        ny = np.clip(ny, AY0+0.1, AY1-0.1)

        # Out of bounds
        out = (ny < AY0) | (ny > AY1)
        self.alive[out] = False

        # Obstacle collisions
        for ox, oy, oz, orx, ory, orz in self.obs:
            hit = (abs(nx-ox) < FISH_R+orx) & (abs(ny-oy) < FISH_R+ory) & (abs(nz-oz) < FISH_R+orz)
            hit = hit & self.alive
            self.energy[hit] -= 10
            self.collisions[hit] += 1
            # Push away
            for i in np.where(hit)[0]:
                dx = nx[i]-ox; dy = ny[i]-oy; dz = nz[i]-oz
                dist = math.hypot(dx, dy, dz)
                if dist > 0.1:
                    push = FISH_R + max(orx, ory, orz)
                    nx[i] += dx/dist*push; ny[i] += dy/dist*push; nz[i] += dz/dist*push
                ny[i] = np.clip(ny[i], AY0+0.1, AY1-0.1)

        # Move non-collided
        move = self.alive.copy()
        self.fx[move] = nx[move]; self.fy[move] = ny[move]; self.fz[move] = nz[move]

        # Eat food
        eaten_idx = []
        for fi, (ffx, ffy, ffz, fr) in enumerate(self.foods):
            for i in np.where(self.alive)[0]:
                d = math.sqrt((self.fx[i]-ffx)**2 + (self.fy[i]-ffy)**2 + (self.fz[i]-ffz)**2)
                if d < fr + 2:
                    self.energy[i] = min(250, self.energy[i] + 60)
                    self.food_eaten[i] += 1
                    eaten_idx.append(fi)
                    break
        for ei in sorted(eaten_idx, reverse=True):
            del self.foods[ei]
        self._sf(len(eaten_idx))

        # Metabolism
        thrust_cost = 0.2*(lt**2 + rt**2 + ut**2 + dt**2)
        self.energy[self.alive] -= 0.8 + thrust_cost[self.alive]
        self.energy = np.clip(self.energy, 0, 300)
        self.alive[(self.energy <= 0)] = False

# ============================================================
# Evolution parameters
# ============================================================
POP = 10; GEN = 15; EPS = 3; STEPS = 150
ELITE = 0.25; SURV = 0.40; MR = 0.03  # mutation rate: 3% replacement
MS = 0.10  # mutation scale: 10% std of pretrained weights

log(f"\nPopulation: {POP}, Generations: {GEN}, Episodes/fish: {EPS}, Steps: {STEPS}")
log(f"Elite: {ELITE:.0%}, Survivor: {SURV:.0%}, Mutation rate: {MR:.0%}, Mutation scale: {MS:.0%}")
log(f"Energy: start=150, basal=-0.8, collision=-10, food=+60, cap=250")

# ============================================================
# Run: pretrained-seeded evolution
# ============================================================
for run_name, init_method in [
    ("v12_pretrained", "pretrained"),
]:
    log(f"\n{'='*60}")
    log(f"RUN: {run_name}")
    log(f"{'='*60}")

    rng = np.random.RandomState(42)
    history = defaultdict(list)

    # Initialize population
    if init_method == "pretrained":
        # Seed from V12 weights + small noise
        noise_scale = MS * np.std(np.abs(v12_bf))
        pop = np.tile(v12_bf, (POP, 1)).astype(np.float32)
        for i in range(POP):
            pop[i] += rng.randn(N_GRU_PARAMS).astype(np.float32) * noise_scale * 0.5
    else:
        # Random Xavier-style init (same as fish_v12.py)
        pop = np.zeros((POP, N_GRU_PARAMS), dtype=np.float32)
        for i in range(POP):
            flat = np.array([], dtype=np.float32)
            for g in ['z', 'r', 'h']:
                for w_name, shape in [('W', (H, GRU_IN)), ('U', (H, H)), ('b', (H,))]:
                    if w_name == 'W':
                        flat = np.concatenate([flat, rng.randn(*shape).astype(np.float32) * math.sqrt(2.0/(H+GRU_IN))])
                    elif w_name == 'U':
                        flat = np.concatenate([flat, rng.randn(*shape).astype(np.float32) * 0.1])
                    else:
                        flat = np.concatenate([flat, np.zeros(shape, dtype=np.float32)])
            flat = np.concatenate([flat, rng.randn(32, H).astype(np.float32).flatten() * math.sqrt(2.0/(32+H))])
            flat = np.concatenate([flat, np.zeros(32, dtype=np.float32)])
            flat = np.concatenate([flat, rng.randn(4, 32).astype(np.float32).flatten() * math.sqrt(2.0/(4+32))])
            flat = np.concatenate([flat, np.zeros(4, dtype=np.float32)])
            pop[i] = flat

    best_all_time = {'gen': -1, 'fitness': -1e9, 'params': None}

    for gen in range(GEN):
        t0 = time.time()
        fitness = np.zeros(POP)

        for ep in range(EPS):
            ep_rng = np.random.RandomState(900000 + gen*1000 + ep)
            arena = Arena(POP, ep_rng)
            arena.reset()
            ht = np.zeros((POP, H), dtype=np.float32)

            for st in range(STEPS):
                alive_mask = arena.alive
                if not alive_mask.any():
                    break

                # Render one fish at a time (DINOv2 is single-sample)
                enc = np.zeros((POP, GRU_IN), dtype=np.float32)
                for i in np.where(alive_mask)[0]:
                    foods_list = [(ffx, ffy, ffz, fr) for ffx, ffy, ffz, fr in arena.foods]
                    L, R = render_lateral(arena.fx[i], arena.fy[i], arena.fz[i], arena.fh[i],
                                          foods_list, arena.obs)
                    enc[i] = retina1408(L, R)

                # Vectorized GRU forward
                out, ht = gru_forward_vectorized(pop, enc, ht)
                lt = out[:, 0]; rt = out[:, 1]; ut = out[:, 2]; dt = out[:, 3]

                # Zero output for dead fish
                lt[~alive_mask] = 0; rt[~alive_mask] = 0
                ut[~alive_mask] = 0; dt[~alive_mask] = 0

                arena.step(lt, rt, ut, dt)

            # Fitness: total food reward (survival bonus implicit via energy)
            fitness += arena.food_eaten * 60  # direct food reward

        fitness /= EPS

        # Sort by fitness
        order = np.argsort(-fitness)
        fitness = fitness[order]
        pop = pop[order]

        log(f"  Gen {gen+1:2d}: bestF={fitness[0]:.0f} meanF={np.mean(fitness):.0f} "
            f"worstF={fitness[-1]:.0f}  ({time.time()-t0:.0f}s)")

        # Update best
        if fitness[0] > best_all_time['fitness']:
            best_all_time = {'gen': gen+1, 'fitness': fitness[0], 'params': pop[0].copy()}

        history['best_fitness'].append(float(fitness[0]))
        history['mean_fitness'].append(float(np.mean(fitness)))
        history['worst_fitness'].append(float(fitness[-1]))

        # Tournament selection
        n_elite = max(1, int(POP * ELITE))
        n_surv = max(2, int(POP * SURV))
        elite = pop[:n_elite].copy()

        # Survivors: tournament
        survivors = []
        tourn_size = 3
        for _ in range(n_surv):
            t_idx = rng.choice(POP, tourn_size, replace=False)
            winner = t_idx[np.argmax(fitness[t_idx])]
            survivors.append(pop[winner].copy())
        survivors = np.array(survivors)

        # Breed: crossover between random pairs of survivors
        n_breed = POP - n_elite
        new_pop = list(elite)
        for _ in range(n_breed):
            p1, p2 = rng.choice(len(survivors), 2, replace=False)
            mask = rng.rand(N_GRU_PARAMS) < 0.5
            child = survivors[p1].copy()
            child[mask] = survivors[p2][mask]
            # Mutation
            mut_mask = rng.rand(N_GRU_PARAMS) < MR
            child[mut_mask] += rng.randn(np.sum(mut_mask)).astype(np.float32) * noise_scale
            new_pop.append(child)
        pop = np.array(new_pop, dtype=np.float32)

    # Save best brain
    best_path = OUT_DIR / f'best_{run_name}.npy'
    np.save(best_path, best_all_time['params'])
    log(f"Best {run_name}: Gen {best_all_time['gen']}, fitness={best_all_time['fitness']:.0f}")
    log(f"Saved: {best_path}")

    # Quick evaluation on a fresh deterministic seed
    log(f"\n  Final evaluation (1 trial, 500 steps, fresh seed):")
    test_rng = np.random.RandomState(12345)
    arena = Arena(1, test_rng)
    arena.reset()
    ht = np.zeros((1, H), dtype=np.float32)
    best_p = best_all_time['params'].reshape(1, -1)
    food_total = 0; collisions = 0

    for st in range(500):
        if not arena.alive[0]:
            break
        foods_list = [(ffx, ffy, ffz, fr) for ffx, ffy, ffz, fr in arena.foods]
        L, R = render_lateral(arena.fx[0], arena.fy[0], arena.fz[0], arena.fh[0],
                              foods_list, arena.obs)
        enc = retina1408(L, R)
        out, ht = gru_forward_vectorized(best_p, enc, ht)
        lt = float(out[0,0]); rt = float(out[0,1]); ut = float(out[0,2]); dt = float(out[0,3])

        te_before = arena.food_eaten[0]
        arena.step([lt], [rt], [ut], [dt])
        if arena.food_eaten[0] > te_before:
            food_total += 1

    collisions = int(arena.collisions[0])
    log(f"  Result: {st+1} steps, {int(arena.food_eaten[0])} food, {collisions} collisions, "
        f"E={arena.energy[0]:.0f}")

    history['final_food'] = int(arena.food_eaten[0])
    history['final_collisions'] = collisions
    history['run_name'] = run_name

    # Save history
    csv_path = OUT_DIR / f'history_{run_name}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gen', 'best_fitness', 'mean_fitness', 'worst_fitness'])
        for g in range(len(history['best_fitness'])):
            w.writerow([g+1, history['best_fitness'][g], history['mean_fitness'][g], history['worst_fitness'][g]])
    log(f"Saved: {csv_path}")

log(f"\n{'='*60}")
log("Done — compare best_v12_pretrained.npy vs best_random_baseline.npy")

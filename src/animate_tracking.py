"""
Dynamic food tracking animation — V12 brain.
- ALL frames rendered (no skip)
- Torus wrap for food (no bounce/jump)
- Food speed < fish max (8 units/step)
- Qualified fish: fish must see food at spawn
- ATE = dist < FOOD_R+2 (same as training/torus)
- Detailed per-fish stats table
"""
import numpy as np, cv2, torch, json, math
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

OUT_DIR = _ROOT / 'animations'
OUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_PATH = _DATA / 'v12_mixed_H128.npy'

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
    """Check if food is in at least one eye's FOV."""
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

def torus_wrap(val, lo, hi, span):
    """Wrap val into [lo, hi] using torus topology."""
    while val > hi: val -= span
    while val < lo: val += span
    return val

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

print(f"Loading: {BRAIN_PATH}")
bf = np.load(BRAIN_PATH)
brain = GRU(); brain.set_params(bf)

FOOD_R = 5.0; FPS = 10
VID_W, VID_H = 1280, 720
MAP_W, MAP_H = 750, 650
MAP_X0, MAP_Y0 = 40, 35
MW = MAP_W/(2*AX); MH = MAP_H/AZ
MAX_STEPS = 350; N_RESETS = 20

# All food speeds < 3 units/step (fish max = 8). Smooth sinusoidal paths.
SCENES = [
    ("orbit",       "Food orbits slowly",
     lambda st: (20*math.sin(st*0.025), 16+5*math.sin(st*0.05), 25+20*math.cos(st*0.025))),
    ("straight",    "Food moves across slowly",
     lambda st: (-24+st*0.20, 18+3*math.sin(st*0.04), 25)),
    ("zigzag",      "Food zigzags in front",
     lambda st: (-18+st*0.12+12*math.sin(st*0.10), 16+4*math.cos(st*0.07), 16+st*0.08)),
    ("figure8",     "Food figure-8",
     lambda st: (14*math.sin(st*0.03), 18+6*math.sin(st*0.06), 22+8*math.cos(st*0.04))),
    ("gentle_sweep","Food gentle sweep",
     lambda st: (20*math.sin(st*0.025), 16+3*math.sin(st*0.04), 22+10*math.cos(st*0.02))),
    ("slow_cross",  "Food crosses slowly",
     lambda st: (-22+st*0.22, 16+5*math.sin(st*0.03), 14+st*0.14)),
]

all_results = []
all_fish_details = []

for si, (scene_name, scene_desc, food_fn) in enumerate(SCENES):
    out_path = str(OUT_DIR / f'track_{si+1:02d}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (VID_W, VID_H))
    total_frame_count = 0; total_eats = 0
    fish_count = 0; qualified_fish = 0; qualified_eats = 0
    scene_fish = []

    fx = np.random.uniform(-0.5*AX, 0.5*AX)
    fy = np.random.uniform(10, 20)
    fz = np.random.uniform(5, 15)
    fh = np.random.uniform(-math.pi, math.pi)
    ht = np.zeros((1, H)); trail = []; food_trail = []

    print(f"\nScene {si+1}/6: {scene_desc}")

    for reset_count in range(N_RESETS):
        # ---- check if fish qualifies (food visible at spawn) ----
        ffx0, ffy0, ffz0 = food_fn(reset_count * MAX_STEPS)
        ffx0 = torus_wrap(ffx0, -AX, AX, 2*AX)
        ffz0 = torus_wrap(ffz0, 0, AZ, AZ)
        ffy0 = np.clip(ffy0, AY0+1, AY1-1)
        fish_can_see = food_visible(fx, fy, fz, fh, ffx0, ffy0, ffz0)
        if fish_can_see:
            qualified_fish += 1
        fish_got_ate = False
        start_dist = None
        end_dist = None
        steps_survived = 0

        for st in range(MAX_STEPS):
            # ---- food position (torus wrap) ----
            ffx, ffy, ffz = food_fn(st + reset_count*MAX_STEPS)
            ffx = torus_wrap(ffx, -AX, AX, 2*AX)
            ffz = torus_wrap(ffz, 0, AZ, AZ)
            ffy = np.clip(ffy, AY0+1, AY1-1)

            out_of_bounds = abs(fx) > AX or fz < 0 or fz > AZ or fy < AY0+0.5 or fy > AY1-0.5
            dist = math.sqrt((fx-ffx)**2 + (fy-ffy)**2 + (fz-ffz)**2)
            just_ate = dist < FOOD_R + 2

            # Record start distance on first step
            if start_dist is None:
                start_dist = dist

            # ---- retina + brain ----
            L, R = render_lateral(fx, fy, fz, fh, [(ffx, ffy, ffz, FOOD_R)])
            enc = retina1408(L, R)
            out, ht = brain.forward(enc, ht)
            lt, rt, ut, dt = float(out[0,0]), float(out[0,1]), float(out[0,2]), float(out[0,3])

            # ---- render frame (EVERY frame, no skip) ----
            map_img = np.zeros((MAP_H, MAP_W, 3), np.uint8); map_img[:] = [25, 22, 18]
            # Grid
            for gx in range(-AX, AX+1, 6):
                px = int(MAP_W/2 + gx*MW)
                cv2.line(map_img, (px, 0), (px, MAP_H), (35, 32, 28), 1)
            for gz in range(0, AZ+1, 10):
                py = int(MAP_H-20 - gz*MH)
                cv2.line(map_img, (0, py), (MAP_W, py), (35, 32, 28), 1)
            cv2.rectangle(map_img, (1, 1), (MAP_W-1, MAP_H-1), (60, 55, 50), 2)
            # Food trail
            food_trail.append((ffx, ffz))
            if len(food_trail) > 60:
                food_trail.pop(0)
            for ti in range(len(food_trail)-1):
                a = ti/max(len(food_trail)-1, 1); c = int(40+160*a)
                p0x = int(MAP_W/2 + food_trail[ti][0]*MW)
                p0y = int(MAP_H-20 - food_trail[ti][1]*MH)
                p1x = int(MAP_W/2 + food_trail[ti+1][0]*MW)
                p1y = int(MAP_H-20 - food_trail[ti+1][1]*MH)
                cv2.line(map_img, (p0x, p0y), (p1x, p1y), (c, c+70, c+50), 2)
            # Food
            fpx = int(MAP_W/2 + ffx*MW); fpy = int(MAP_H-20 - ffz*MH)
            fpr = max(6, int(FOOD_R*MW))
            cv2.circle(map_img, (fpx, fpy), fpr, (0, 240, 100), -1)
            cv2.circle(map_img, (fpx, fpy), fpr, (0, 150, 50), 2)
            cv2.putText(map_img, "FOOD", (fpx+fpr+3, fpy+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 240, 100), 1)
            # Fish
            fish_px = int(MAP_W/2 + fx*MW); fish_py = int(MAP_H-20 - fz*MH)
            fl_w, fw_w = 10, 6
            nose = (int(fish_px + fl_w*math.sin(fh)), int(fish_py - fl_w*math.cos(fh)))
            tl_w = (int(fish_px + fw_w*math.cos(fh)), int(fish_py + fw_w*math.sin(fh)))
            tr_w = (int(fish_px - fw_w*math.cos(fh)), int(fish_py + fw_w*math.sin(fh)))
            cv2.fillPoly(map_img, [np.array([nose, tl_w, tr_w])], (80, 180, 255))
            cv2.polylines(map_img, [np.array([nose, tl_w, tr_w])], True, (180, 230, 255), 2)
            # Fish trail
            trail.append((fish_px, fish_py))
            if len(trail) > 80:
                trail.pop(0)
            for ti in range(len(trail)-1):
                a = ti/max(len(trail)-1, 1); c = int(40+160*a)
                cv2.line(map_img, trail[ti], trail[ti+1], (c, c+50, c+70), 1)
            # Info text
            cv2.putText(map_img,
                        f"Fish({fx:.0f},{fz:.0f}) h={math.degrees(fh):.0f}",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(map_img,
                        f"D:{dist:.0f} Q:{qualified_eats}/{qualified_fish} E:{total_eats}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(map_img, scene_desc,
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            if just_ate:
                cv2.putText(map_img, "ATE!",
                            (MAP_W//2-60, MAP_H//2-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 100), 6)
            # Compose canvas
            canvas = np.zeros((VID_H, VID_W, 3), np.uint8); canvas[:] = [22, 20, 18]
            canvas[MAP_Y0:MAP_Y0+MAP_H, MAP_X0:MAP_X0+MAP_W] = map_img
            rx = MAP_X0 + MAP_W + 15; ew, eh = 230, 148; gap = 4
            cv2.putText(canvas, "V12 TRACKING", (rx, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 120), 1)
            l_img = cv2.resize(L, (ew, eh))
            canvas[30:30+eh, rx:rx+ew] = l_img
            cv2.rectangle(canvas, (rx, 30), (rx+ew, 30+eh), (70, 70, 70), 1)
            ly = 30 + eh + gap
            r_img = cv2.resize(R, (ew, eh))
            canvas[ly:ly+eh, rx:rx+ew] = r_img
            cv2.rectangle(canvas, (rx, ly), (rx+ew, ly+eh), (70, 70, 70), 1)
            dy = ly + eh + gap
            stereo = cv2.resize(cv2.applyColorMap(cv2.absdiff(L, R), cv2.COLORMAP_HOT), (ew, eh))
            canvas[dy:dy+eh, rx:rx+ew] = stereo
            cv2.rectangle(canvas, (rx, dy), (rx+ew, dy+eh), (70, 70, 70), 1)
            sy = dy + eh + 12
            for li, (l, v) in enumerate([("L", f"{lt:+.2f}"), ("R", f"{rt:+.2f}"),
                                          ("D", f"{dist:.0f}")]):
                cv2.putText(canvas, f"{l}:{v}", (rx+5, sy+li*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            writer.write(canvas)
            total_frame_count += 1

            # ---- move fish ----
            fwd = (lt+rt)/2 * 8
            fh += (rt-lt) * 0.5
            fx = fx + fwd*math.sin(fh)
            fz = fz + fwd*math.cos(fh)
            fy = fy + (ut-dt) * 5

            end_dist = dist
            steps_survived = st + 1

            # ---- termination ----
            if just_ate or out_of_bounds or st == MAX_STEPS-1:
                if just_ate:
                    total_eats += 1
                    fish_got_ate = True
                fish_count += 1
                if fish_can_see and fish_got_ate:
                    qualified_eats += 1

                # Record per-fish stats
                scene_fish.append({
                    'idx': fish_count,
                    'qualified': fish_can_see,
                    'start_dist': round(start_dist, 1) if start_dist else None,
                    'end_dist': round(end_dist, 1) if end_dist else None,
                    'ate': fish_got_ate,
                    'steps': steps_survived,
                })

                # Respawn fish
                fx = np.random.uniform(-0.5*AX, 0.5*AX)
                fy = np.random.uniform(10, 20)
                fz = np.random.uniform(5, 15)
                fh = np.random.uniform(-math.pi, math.pi)
                ht = np.zeros((1, H))
                trail = []
                break

    writer.release()
    qef = qualified_eats / max(1, qualified_fish)
    all_results.append((scene_name, scene_desc, total_eats, fish_count,
                        qualified_eats, qualified_fish, qef))
    all_fish_details.append(scene_fish)
    print(f"  Saved: {out_path}")
    print(f"  Q:{qualified_eats}/{qualified_fish}={qef:.2f}  All:{total_eats}/{fish_count}={total_eats/max(1,fish_count):.2f}")

# ============================================================
# Detailed statistics
# ============================================================
print(f"\n{'='*80}")
print("DYNAMIC TRACKING — V12 Brain — ATE metric")
print(f"{'='*80}")

for si, (scene_name, scene_desc, te, fc, qe, qf, qef) in enumerate(all_results):
    print(f"\n--- Scene {si+1}: {scene_name} ({scene_desc}) ---")
    print(f"    Summary: Q={qe}/{qf}={qef:.2f}  All={te}/{fc}={te/max(1,fc):.2f}")
    fd = all_fish_details[si]
    print(f"    {'#':>3s} {'Qual?':>5s} {'StartD':>7s} {'EndD':>7s} {'Ate?':>5s} {'Steps':>5s}")
    for f in fd:
        qstr = "YES" if f['qualified'] else "no"
        sstr = f"{f['start_dist']:.0f}" if f['start_dist'] else "?"
        estr = f"{f['end_dist']:.0f}" if f['end_dist'] else "?"
        astr = "ATE!" if f['ate'] else "-"
        print(f"    {f['idx']:>3d} {qstr:>5s} {sstr:>7s} {estr:>7s} {astr:>5s} {f['steps']:>5d}")

# Grand total
tq = sum(r[4] for r in all_results)
tfq = sum(r[5] for r in all_results)
tall_e = sum(r[2] for r in all_results)
tall_f = sum(r[3] for r in all_results)
print(f"\n{'='*80}")
print(f"TOTAL: Qualified={tq}/{tfq}={tq/max(1,tfq):.2f}  AllEats={tall_e}/{tall_f}={tall_e/max(1,tall_f):.2f}")
print("Q = Qualified: fish that could SEE food at spawn (ATE metric = standard)")
print("All food speeds < 3 units/step (fish max = 8)")
print("DONE")

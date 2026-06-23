"""
Phase 3: Weight Pathway Analysis
Trace W2(4×32) × W1(32×128) to find which hidden dimensions drive which outputs.
Also analyze U_z, U_h recurrent connections to see functional clustering.

Usage: python weight_pathway.py
"""
import numpy as np, json, csv
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = _ROOT / 'circuit_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

H = 128; GRU_IN = 1408

def load_weights(brain_path):
    bf = np.load(brain_path)
    idx = 0
    weights = {}
    for g in ['z', 'r', 'h']:
        for p in ['W', 'U', 'b']:
            name = f'{p}_{g}'
            if p == 'W':
                shape = (H, GRU_IN)
            elif p == 'U':
                shape = (H, H)
            else:
                shape = (H,)
            a = np.zeros(shape)
            m = a.size
            a.flat = bf[idx:idx+m]
            idx += m
            weights[name] = a
    for name, shape in [('W1', (32, H)), ('b1', (32,)), ('W2', (4, 32)), ('b2', (4,))]:
        a = np.zeros(shape)
        m = a.size
        a.flat = bf[idx:idx+m]
        idx += m
        weights[name] = a
    return weights

BRAIN_PATH = _DATA / 'v12_mixed_H128.npy'
w = load_weights(BRAIN_PATH)
print(f"Loaded weights from {BRAIN_PATH.name}")

# ============================================================
# 1. Effective hidden→output pathway
# ============================================================
# After ReLU(h @ W1.T + b1), the mid layer is 32-dim, then tanh(mid @ W2.T + b2)
# Effective linear: W_eff = W2 @ W1  shape (4, 128)
W_eff = w['W2'] @ w['W1']  # 4 × 128

# L, R, U, D contributions
L_w = W_eff[0]; R_w = W_eff[1]; U_w = W_eff[2]; D_w = W_eff[3]
fwd_w = (L_w + R_w) / 2   # forward thrust: average of L and R
turn_w = (R_w - L_w)      # turn: difference between R and L
vert_w = (U_w - D_w)      # vertical: difference between U and D

# Sort hidden dims by contribution
turn_order = np.argsort(-np.abs(turn_w))   # largest |turn| contribution first
fwd_order = np.argsort(-np.abs(fwd_w))     # largest |fwd| contribution first
vert_order = np.argsort(-np.abs(vert_w))   # largest |vert| contribution first

print(f"\n--- Top 16 hidden dims by |turn| contribution ---")
print(f"  Dim  turn_w    fwd_w    vert_w")
for i in range(16):
    d = turn_order[i]
    print(f"  {d:3d}  {turn_w[d]:+8.4f}  {fwd_w[d]:+8.4f}  {vert_w[d]:+8.4f}")

print(f"\n--- Top 16 hidden dims by |fwd| contribution ---")
print(f"  Dim  fwd_w     turn_w    vert_w")
for i in range(16):
    d = fwd_order[i]
    print(f"  {d:3d}  {fwd_w[d]:+8.4f}  {turn_w[d]:+8.4f}  {vert_w[d]:+8.4f}")

# Overlap between turn and fwd dims
top16_turn = set(turn_order[:16])
top16_fwd = set(fwd_order[:16])
overlap = top16_turn & top16_fwd
print(f"\nOverlap between top-16 turn and top-16 fwd: {len(overlap)} dims: {sorted(overlap)}")

# ============================================================
# 2. Functional specialization index
# ============================================================
# For each dim: specialization = |turn_w| / max(|fwd_w|, 1e-6)
# High value → dim is turn-specialized; low → fwd-specialized
spec = np.abs(turn_w) / (np.abs(fwd_w) + 1e-8)
turn_specialized = np.argsort(-spec)  # highest turn/fwd ratio
fwd_specialized = np.argsort(spec)    # lowest turn/fwd ratio

print(f"\n--- Turn-specialized dims (high |turn|/|fwd|) ---")
for i in range(12):
    d = turn_specialized[i]
    print(f"  dim {d:3d}: turn={turn_w[d]:+.4f} fwd={fwd_w[d]:+.4f} ratio={spec[d]:.1f}")

print(f"\n--- Fwd-specialized dims (low |turn|/|fwd|) ---")
for i in range(12):
    d = fwd_specialized[i]
    print(f"  dim {d:3d}: turn={turn_w[d]:+.4f} fwd={fwd_w[d]:+.4f} ratio={spec[d]:.3f}")

# ============================================================
# 3. Recurrent weight structure (U_z, U_h)
# ============================================================
# Check if turn-specialized dims connect to each other in U_z and U_h
U_z = w['U_z']  # 128 × 128
U_h = w['U_h']  # 128 × 128

# Block analysis: connectivity within turn-dims vs fwd-dims vs cross
turn16 = list(turn_specialized[:16])
fwd16 = list(fwd_specialized[:16])

u_z_turn_turn = np.mean(np.abs(U_z[turn16][:, turn16]))
u_z_fwd_fwd = np.mean(np.abs(U_z[fwd16][:, fwd16]))
u_z_turn_fwd = np.mean(np.abs(U_z[turn16][:, fwd16]))
u_z_fwd_turn = np.mean(np.abs(U_z[fwd16][:, turn16]))

print(f"\n--- U_z recurrent connectivity ---")
print(f"  turn→turn: {u_z_turn_turn:.6f}")
print(f"  fwd→fwd:   {u_z_fwd_fwd:.6f}")
print(f"  turn→fwd:  {u_z_turn_fwd:.6f}")
print(f"  fwd→turn:  {u_z_fwd_turn:.6f}")

u_h_turn_turn = np.mean(np.abs(U_h[turn16][:, turn16]))
u_h_fwd_fwd = np.mean(np.abs(U_h[fwd16][:, fwd16]))
u_h_turn_fwd = np.mean(np.abs(U_h[turn16][:, fwd16]))
u_h_fwd_turn = np.mean(np.abs(U_h[fwd16][:, turn16]))

print(f"\n--- U_h recurrent connectivity ---")
print(f"  turn→turn: {u_h_turn_turn:.6f}")
print(f"  fwd→fwd:   {u_h_fwd_fwd:.6f}")
print(f"  turn→fwd:  {u_h_turn_fwd:.6f}")
print(f"  fwd→turn:  {u_h_fwd_turn:.6f}")

# ============================================================
# 4. Input weight analysis: which input features drive turn/fwd dims?
# ============================================================
# W_z: 128 × 1408, W_h: 128 × 1408
# Input partitioning: cL(0:384), cR(384:768), disp(768:1152), micro(1152:1408)
segments = {'cL': (0, 384), 'cR': (384, 768), 'disp': (768, 1152), 'micro': (1152, 1408)}

print(f"\n--- Input feature sensitivity (W_z) for turn vs fwd dims ---")
for dimset, label in [(turn16, 'turn-spec'), (fwd16, 'fwd-spec'), (list(range(128)), 'all')]:
    W_z_sub = w['W_z'][dimset]  # N × 1408
    for seg_name, (s0, s1) in segments.items():
        seg_norm = np.mean(np.abs(W_z_sub[:, s0:s1]))
        if label == 'all':
            print(f"  {label:12s} → {seg_name:6s}: {seg_norm:.6f}")
        else:
            print(f"  {label:12s} → {seg_name:6s}: {seg_norm:.6f}")

print(f"\n--- Input feature sensitivity (W_h) for turn vs fwd dims ---")
for dimset, label in [(turn16, 'turn-spec'), (fwd16, 'fwd-spec'), (list(range(128)), 'all')]:
    W_h_sub = w['W_h'][dimset]
    for seg_name, (s0, s1) in segments.items():
        seg_norm = np.mean(np.abs(W_h_sub[:, s0:s1]))
        if label == 'all':
            print(f"  {label:12s} → {seg_name:6s}: {seg_norm:.6f}")
        else:
            print(f"  {label:12s} → {seg_name:6s}: {seg_norm:.6f}")

# ============================================================
# 5. Plot
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 5a. W_eff heatmap (4 × 128)
ax = axes[0, 0]
im = ax.imshow(W_eff, aspect='auto', cmap='RdBu_r', vmin=-np.max(np.abs(W_eff)), vmax=np.max(np.abs(W_eff)))
ax.set_yticks([0,1,2,3]); ax.set_yticklabels(['L','R','U','D'])
ax.set_xlabel('Hidden dimension'); ax.set_title('Effective Output Weights (W2 @ W1)')
plt.colorbar(im, ax=ax)

# 5b. Turn vs fwd weight scatter
ax = axes[0, 1]
sc = ax.scatter(fwd_w, turn_w, c=np.arange(128), cmap='viridis', s=30, alpha=0.8)
for i in range(128):
    if i in turn16:
        ax.annotate(str(i), (fwd_w[i], turn_w[i]), fontsize=6, color='red', fontweight='bold')
ax.axhline(y=0, color='gray', ls='--', lw=0.5)
ax.axvline(x=0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('fwd weight (L+R)/2'); ax.set_ylabel('turn weight (R-L)')
ax.set_title('Hidden Dims: Turn vs Fwd Contribution')
plt.colorbar(sc, ax=ax, label='dim index')

# 5c. Specialization ratio bar
ax = axes[0, 2]
spec_sorted = np.sort(spec)[::-1]
ax.bar(range(128), spec_sorted, color=['#d62728' if s > 3 else '#1f77b4' if s < 0.3 else '#7f7f7f' for s in spec_sorted])
ax.set_xlabel('Hidden dim (sorted by turn/fwd ratio)')
ax.set_ylabel('|turn| / |fwd| ratio')
ax.set_title('Functional Specialization per Hidden Dim')

# 5d. U_z matrix with turn/fwd blocks
ax = axes[1, 0]
# Reorder dims: turn-spec first, then fwd-spec, then rest
reorder = list(turn_specialized[:16]) + list(fwd_specialized[:16]) + \
          [d for d in range(128) if d not in turn_specialized[:16] and d not in fwd_specialized[:16]]
U_z_reordered = U_z[reorder][:, reorder]
ax.imshow(U_z_reordered, aspect='auto', cmap='RdBu_r',
          vmin=-np.max(np.abs(U_z_reordered)), vmax=np.max(np.abs(U_z_reordered)))
ax.axhline(y=15.5, color='lime', lw=1.5)
ax.axvline(x=15.5, color='lime', lw=1.5)
ax.axhline(y=31.5, color='orange', lw=1.5)
ax.axvline(x=31.5, color='orange', lw=1.5)
ax.set_title('U_z (reordered: turn16 | fwd16 | rest)')

# 5e. U_h matrix
ax = axes[1, 1]
U_h_reordered = U_h[reorder][:, reorder]
ax.imshow(U_h_reordered, aspect='auto', cmap='RdBu_r',
          vmin=-np.max(np.abs(U_h_reordered)), vmax=np.max(np.abs(U_h_reordered)))
ax.axhline(y=15.5, color='lime', lw=1.5)
ax.axvline(x=15.5, color='lime', lw=1.5)
ax.axhline(y=31.5, color='orange', lw=1.5)
ax.axvline(x=31.5, color='orange', lw=1.5)
ax.set_title('U_h (reordered: turn16 | fwd16 | rest)')

# 5f. Input feature preference: turn vs fwd dims
ax = axes[1, 2]
x = np.arange(4); width = 0.25
for di, (dimset, label, color) in enumerate([
    (turn16, 'turn-spec', '#d62728'),
    (fwd16, 'fwd-spec', '#1f77b4'),
    (list(range(128)), 'all dims', '#7f7f7f'),
]):
    means = [np.mean(np.abs(w['W_h'][dimset][:, s0:s1])) for s0,s1 in segments.values()]
    ax.bar(x + di*width, means, width, color=color, alpha=0.7, label=label)
ax.set_xticks(x + width); ax.set_xticklabels(segments.keys())
ax.set_ylabel('mean |W_h| weight'); ax.set_title('Input Feature Sensitivity (W_h)')
ax.legend(fontsize=8)

plt.tight_layout()
fig_path = OUT_DIR / 'weight_pathway.png'
fig.savefig(fig_path, dpi=120)
print(f"\nSaved: {fig_path}")

# ============================================================
# Save CSV
# ============================================================
csv_path = OUT_DIR / 'pathway_weights.csv'
with open(csv_path, 'w', newline='') as f:
    w_csv = csv.writer(f)
    w_csv.writerow(['dim','L_weight','R_weight','U_weight','D_weight',
                    'fwd_weight','turn_weight','vert_weight',
                    'turn_fwd_ratio','is_turn_spec','is_fwd_spec'])
    for d in range(128):
        w_csv.writerow([d, L_w[d], R_w[d], U_w[d], D_w[d],
                        fwd_w[d], turn_w[d], vert_w[d],
                        round(spec[d], 3),
                        1 if d in turn16 else 0,
                        1 if d in fwd16 else 0])
print(f"Saved: {csv_path}")

# Summary
print(f"\n--- SUMMARY ---")
print(f"Turn-specialized dims (ratio>3): {list(turn_specialized[:16])}")
print(f"Fwd-specialized dims (ratio<0.3): {list(fwd_specialized[:16])}")
print(f"Overlap top-16 turn/fwd: {overlap}")
print("DONE — Phase 3")

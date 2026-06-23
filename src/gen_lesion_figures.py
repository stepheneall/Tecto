"""Generate publication figures for the lesion study."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = _ROOT / 'figures_pub'
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
    'legend.fontsize':8,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight'})

# === FIGURE: b_z dose-response curve ===
print("Figure: b_z dose-response...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

deltas = [-0.50, -0.20, -0.10, 0.00, 0.10, 0.20, 0.50]
z_actual = [0.405, 0.477, 0.502, 0.526, 0.551, 0.575, 0.646]
turn_std = [0.055, 0.057, 0.051, 0.052, 0.054, 0.064, 0.070]
h_osc = [0.134, 0.127, 0.122, 0.120, 0.114, 0.110, 0.096]

# Panel A: z tracks Delta linearly
ax1.plot(deltas, z_actual, 'o-', color='#1f77b4', lw=2, markersize=8)
ax1.axhline(y=0.526, color='gray', ls='--', lw=0.8, alpha=0.5)
ax1.axvline(x=0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax1.fill_between([-0.5, 0], 0.38, 0.65, alpha=0.05, color='#d62728')
ax1.fill_between([0, 0.5], 0.38, 0.65, alpha=0.05, color='#2ca02c')
ax1.text(-0.25, 0.42, 'Bradykinesia', fontsize=7, ha='center', color='#d62728')
ax1.text(0.25, 0.62, 'Hyperkinesia', fontsize=7, ha='center', color='#2ca02c')
ax1.set_xlabel(r'$\Delta$ (b$_z$ shift)'); ax1.set_ylabel('z actual')
ax1.set_title('A: Gain tracks shift linearly')

# Panel B: ATE invariant
ax2.bar(range(len(deltas)), [1.0]*7, color='#1f77b4', alpha=0.5, edgecolor='#333', linewidth=0.5)
ax2.set_xticks(range(len(deltas)))
ax2.set_xticklabels([f'{d:+.1f}' for d in deltas], fontsize=8)
ax2.set_ylabel('ATE'); ax2.set_ylim(0, 1.2)
ax2.set_title('B: ATE invariant (1.00 at all gain levels)')
ax2.axhline(y=1.0, color='#2ca02c', lw=2, ls='--', alpha=0.5)

# Panel C: turn_std vs h_osc tradeoff
ax3_t = ax3
color = '#d62728'
ax3_t.plot(deltas, turn_std, 'o-', color='#d62728', lw=2, markersize=8, label='turn_std')
ax3_t.set_xlabel(r'$\Delta$ (b$_z$ shift)'); ax3_t.set_ylabel('turn_std', color='#d62728')
ax3_h = ax3.twinx()
ax3_h.plot(deltas, h_osc, 's-', color='#1f77b4', lw=2, markersize=8, label='h_osc')
ax3_h.set_ylabel('h_osc', color='#1f77b4')
ax3.set_title('C: Sharper turns dampen oscillation')
lines1, labels1 = ax3_t.get_legend_handles_labels()
lines2, labels2 = ax3_h.get_legend_handles_labels()
ax3.legend(lines1+lines2, labels1+labels2, fontsize=7)

fig.suptitle('Gain Control Dose-Response (b$_z$ shift)', fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_lesion_bz_doseresponse.png', dpi=300)
plt.close()
print("  Saved: fig_lesion_bz_doseresponse.png")

# === FIGURE: W2 row ablation ===
print("Figure: W2 row ablation...")
fig, ax = plt.subplots(figsize=(10, 5))
conditions = ['Intact', 'no_L\nthrust', 'no_R\nthrust', 'half_L', 'half_R', 'no_H']
mean_turns = [-0.111, +0.475, -0.655, +0.043, -0.271, -0.069]
turn_stds = [0.648, 0.371, 0.345, 0.591, 0.588, 0.000]
colors = ['#7f7f7f', '#d62728', '#1f77b4', '#ff9896', '#aec7e8', '#2ca02c']
x = np.arange(len(conditions))
bars = ax.bar(x, mean_turns, color=colors, edgecolor='#333', linewidth=0.5)
ax.axhline(y=0, color='gray', lw=0.8, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=9)
ax.set_ylabel('Mean Turn (R-L)')
ax.set_title('W2 Row Ablation — Direction-Specific Motor Deficits', fontweight='bold')
# Annotate
for i, (bar, mt, ts) in enumerate(zip(bars, mean_turns, turn_stds)):
    ax.text(bar.get_x()+bar.get_width()/2, mt+0.05*np.sign(mt) if abs(mt)>0.05 else mt+0.05,
            f'{mt:+.3f}\nσ={ts:.3f}', ha='center', fontsize=7, fontweight='bold')
ax.text(1, 0.8, 'Right\nbias', ha='center', fontsize=9, fontweight='bold', color='#d62728')
ax.text(2, -0.9, 'Left\nbias', ha='center', fontsize=9, fontweight='bold', color='#1f77b4')
ax.set_ylim(-1.0, 1.0)
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_lesion_w2_ablation.png', dpi=300)
plt.close()
print("  Saved: fig_lesion_w2_ablation.png")

# === FIGURE: Frequency resonance ===
print("Figure: Frequency resonance...")
fig, ax = plt.subplots(figsize=(9, 5.5))
freqs = [1, 2, 3, 5, 8, 12, 16, 24, 30]
wh59 = [0.321, 0.308, 0.297, 0.244, 0.208, 0.158, 0.127, 0.102, 0.104]
wz59 = [0.043, 0.041, 0.039, 0.033, 0.026, 0.021, 0.017, 0.015, 0.014]
rand = [0.031, 0.032, 0.036, 0.031, 0.028, 0.030, 0.032, 0.034, 0.034]

ax.plot(freqs, wh59, 'o-', color='#d62728', lw=2.5, markersize=10, label=r'W$_h$ col 59 (sensory)')
ax.plot(freqs, wz59, 's-', color='#1f77b4', lw=2, markersize=8, label=r'W$_z$ col 59 (gain)')
ax.plot(freqs, rand, 'D-', color='#7f7f7f', lw=1.5, markersize=6, label='Random vector')
ax.axvspan(1, 5, alpha=0.08, color='#d62728')
ax.text(3, 0.34, r'$\delta$/$\theta$ band', fontsize=9, ha='center', color='#d62728', fontweight='bold')

# Add ratio annotation
ax.annotate('10x attenuation\nvs. sensory pathway', xy=(8, 0.028), xytext=(14, 0.15),
            arrowprops=dict(arrowstyle='->', color='#7f7f7f'), fontsize=8, color='#7f7f7f')
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Hidden-state oscillation amplitude')
ax.set_title('Sensory Pathway Frequency Resonance', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(0, 32); ax.set_ylim(0, 0.38)
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_lesion_freq_resonance.png', dpi=300)
plt.close()
print("  Saved: fig_lesion_freq_resonance.png")

print("Done.")

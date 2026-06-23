"""Generate all publication-quality figures for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

# FIG 1: Per-class mAP vs Thick-Scene Count
classes = ['Person', 'Labcoat', 'Uniform']
mAP = [0.94, 0.87, 0.58]
thick_scenes = [116, 110, 4]
colors = ['#4CAF50','#2196F3','#FF5722']
x = np.arange(3)
fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x, mAP, 0.5, color=colors, edgecolor='white', linewidth=1.5)
for i,(v,ts) in enumerate(zip(mAP, thick_scenes)):
    ax.text(i, v+0.02, f'mAP={v:.2f}\n{ts} thick scenes', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=13)
ax.set_ylabel('Test mAP@0.5', fontsize=12)
ax.set_title('Per-Class Detection Performance vs. Thick-Scene Count', fontsize=14, fontweight='bold')
ax.set_ylim(0,1.05); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('fig_class_scene_gap.pdf', dpi=200, bbox_inches='tight')
print('fig_class_scene_gap.pdf')

# FIG 2: Uniform per-scene instance histogram
labels = ['1','2','3','4-5','6-7','8-10','11-15','16-20','21-30','31-50','51+']
counts = [34,23,26,15,9,12,7,4,2,2,3]
cum_pct = np.cumsum(counts)/sum(counts)*100
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(range(len(labels)), counts, color='#FF5722', edgecolor='white', linewidth=1)
for i,(l,c) in enumerate(zip(labels,counts)):
    ax.text(i, c+1, str(c), ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Number of scenes', fontsize=12)
ax.set_xlabel('Uniform instances per scene', fontsize=12)
ax.set_title('Uniform Class: Per-Scene Instance Distribution', fontsize=14, fontweight='bold')
ax2 = ax.twinx()
ax2.plot(range(len(labels)), cum_pct, 'o-', color='#1565C0', linewidth=2, markersize=8)
ax2.set_ylabel('Cumulative % of instances', fontsize=12, color='#1565C0')
ax2.tick_params(axis='y', labelcolor='#1565C0')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('fig_uniform_hist.pdf', dpi=200, bbox_inches='tight')
print('fig_uniform_hist.pdf')

# FIG 3: PCA comparison
coco_pcs = [99.5,0.4,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
dino_pcs = [69.9,4.7,3.4,2.4,2.0,1.8,1.6,1.4,1.2,1.0]
xs = list(range(1,11))
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4.5))
ax1.bar(xs, coco_pcs, color='#E91E63', edgecolor='white')
ax1.set_title('COCO Backbone (SPPF)', fontsize=12, fontweight='bold')
ax1.set_ylabel('% Variance', fontsize=11); ax1.set_xlabel('Principal Component')
ax1.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
ax2.bar(xs, dino_pcs, color='#1565C0', edgecolor='white')
ax2.set_title('DINOv2 ViT-B/14 (CLS)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Principal Component')
ax2.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
for ax in [ax1,ax2]: ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.suptitle('PCA Variance: Detection Backbone vs. Self-Supervised Encoder', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('fig_pca_comparison.pdf', dpi=200, bbox_inches='tight')
print('fig_pca_comparison.pdf')

# FIG 4: Distance distribution histogram
D = np.load('dinov2_distance.npy')
n = len(D); mask = ~np.eye(n,dtype=bool); d_flat = D[mask]
fig, ax = plt.subplots(figsize=(8,4.5))
ax.hist(d_flat, bins=50, color='#1565C0', alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(x=0.15, color='#E91E63', linestyle='--', linewidth=2, label='r=0.15 (cover radius)')
ax.axvline(x=np.median(d_flat), color='#4CAF50', linestyle=':', linewidth=2, label=f'median={np.median(d_flat):.3f}')
ax.set_xlabel('DINOv2 Cosine Distance', fontsize=12)
ax.set_ylabel('Number of scene pairs', fontsize=12)
ax.set_title('Pairwise DINOv2 Distance Distribution (159 scenes)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('fig_dist_hist.pdf', dpi=200, bbox_inches='tight')
print('fig_dist_hist.pdf')

# FIG 5: Cover vs Anti-cover internal distances
cov_dict = {i:set(j for j in range(n) if D[i][j]<=0.15) for i in range(n)}
rem, sel = set(range(n)), []
while rem:
    b, bn = None, 0
    for i in range(n):
        nw = len(cov_dict[i]&rem)
        if nw>bn: bn,b=nw,i
    if bn==0: break; sel.append(b); rem-=cov_dict[b]
c_idx=set(sel); a_idx=set(range(n))-c_idx
c_dist=[D[i][j] for i in c_idx for j in c_idx if i<j]
a_dist=[D[i][j] for i in a_idx for j in a_idx if i<j]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5))
ax1.hist(c_dist,bins=40,color='#2196F3',alpha=0.7,edgecolor='white')
ax1.axvline(x=np.mean(c_dist),color='red',linestyle='--',label=f'mean={np.mean(c_dist):.3f}')
ax1.set_title(f'Cover Set (96 scenes)\nmean={np.mean(c_dist):.3f} median={np.median(c_dist):.3f}',fontsize=12,fontweight='bold')
ax1.set_xlabel('DINOv2 cosine distance'); ax1.legend()
ax2.hist(a_dist,bins=40,color='#FF5722',alpha=0.7,edgecolor='white')
ax2.axvline(x=np.mean(a_dist),color='red',linestyle='--',label=f'mean={np.mean(a_dist):.3f}')
ax2.set_title(f'Anti-Cover Set (63 scenes)\nmean={np.mean(a_dist):.3f} median={np.median(a_dist):.3f}',fontsize=12,fontweight='bold')
ax2.set_xlabel('DINOv2 cosine distance'); ax2.legend()
for ax in [ax1,ax2]: ax.grid(axis='y',alpha=0.3)
plt.suptitle('Internal Distance Distribution: Cover vs. Anti-Cover Sets',fontsize=13,fontweight='bold')
plt.tight_layout(); plt.savefig('fig_internal_dists.pdf',dpi=200,bbox_inches='tight')
print('fig_internal_dists.pdf')

# FIG 6: Full-159 vs A_cover bar
fig,ax = plt.subplots(figsize=(6,4.5))
ax.bar([0,1],[0.931,0.963],0.4,color=['#2196F3','#4CAF50'],edgecolor='white',linewidth=2)
ax.text(0,0.941,'0.931',ha='center',fontsize=14,fontweight='bold')
ax.text(1,0.973,'0.963',ha='center',fontsize=14,fontweight='bold')
ax.set_xticks([0,1]); ax.set_xticklabels(['A_cover (96, zero-shot)','Full-159 (159, anchor-trained)'],fontsize=11)
ax.set_ylabel('Recall on Anti-Cover Scenes',fontsize=11)
ax.set_ylim(0.88,1.0); ax.axhline(y=0.963,color='#4CAF50',linestyle='--',alpha=0.3)
ax.set_title('Sparse vs. Full Anchoring on Anti-Cover Scenes',fontsize=14,fontweight='bold')
ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig('fig_full159_bar.pdf',dpi=200,bbox_inches='tight')
print('fig_full159_bar.pdf')

# FIG 7: Multi-r trade-off curve
r_vals=[0.12,0.15,0.18,0.21,0.24,0.27]
K_vals=[134,96,70,43,31,20]
anchor_pct=[k/159*100 for k in K_vals]
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(r_vals,anchor_pct,'o-',color='#E91E63',linewidth=2.5,markersize=10,markerfacecolor='white')
for r,k,p in zip(r_vals,K_vals,anchor_pct):
    ax.annotate(f'K={k}\n({p:.0f}%)',(r,p),textcoords='offset points',xytext=(0,15),ha='center',fontsize=9)
ax.set_xlabel('Coverage Radius r (DINOv2 cosine distance)',fontsize=12)
ax.set_ylabel('Anchor scenes required (% of total)',fontsize=12)
ax.set_title('Coverage-Cost Trade-Off: Anchors vs. Radius',fontsize=14,fontweight='bold')
ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('fig_multir_tradeoff.pdf',dpi=200,bbox_inches='tight')
print('fig_multir_tradeoff.pdf')

# FIG 8: Scaling projection
N_pred=[100,200,500,1000,2000,5000,10000]
fig,ax=plt.subplots(figsize=(7,5))
for r,b,a,color in [(0.15,0.844,1.36,'#1565C0'),(0.18,0.781,1.35,'#4CAF50'),(0.21,0.686,1.39,'#FF5722'),(0.27,0.588,1.02,'#9C27B0')]:
    K_pred=[a*n**b for n in N_pred]
    pct_pred=[k/n*100 for k,n in zip(K_pred,N_pred)]
    ax.plot(N_pred,pct_pred,'o-',color=color,linewidth=2,markersize=8,label=f'r={r:.2f} ($K \\propto N^{{{b:.2f}}}$)')
ax.set_xscale('log'); ax.set_xlabel('Total cameras N',fontsize=12)
ax.set_ylabel('Anchors required (%)',fontsize=12)
ax.set_title('Scaling: Anchor Ratio vs. Domain Size',fontsize=14,fontweight='bold')
ax.legend(fontsize=9,loc='upper right'); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('fig_scaling.pdf',dpi=200,bbox_inches='tight')
print('fig_scaling.pdf')

# FIG 9: Cover set size vs r
fig,ax=plt.subplots(figsize=(6,4.5))
ax.plot(r_vals,K_vals,'s-',color='#E91E63',linewidth=2.5,markersize=10,markerfacecolor='white')
for r,k in zip(r_vals,K_vals):
    ax.annotate(f'{k}',(r,k),textcoords='offset points',xytext=(0,10),ha='center',fontsize=10,fontweight='bold')
ax.set_xlabel('Coverage radius r',fontsize=12)
ax.set_ylabel('Cover set size K',fontsize=12)
ax.set_title('Cover Set Size vs. Coverage Radius',fontsize=14,fontweight='bold')
ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('fig_cover_size.pdf',dpi=200,bbox_inches='tight')
print('fig_cover_size.pdf')

print('All 9 data figures generated.')

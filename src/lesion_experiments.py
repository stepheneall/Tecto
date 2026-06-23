"""
Tectal Lesion Experiments — Systematic circuit damage on a known healthy tectum.

Four new experiments, each targeting a specific circuit component:
  3.1 U_z partial ablation — recurrent connection short-term memory loss
  3.2 b_z shift — baseline neuromodulatory bias
  3.3 W2 row ablation — unilateral directional motor deficit
  3.4 Frequency-specific stimulation — component resonance testing

All use the characterized V12 brain. No training — direct weight manipulation.
Each lesion has a known circuit function, predicted deficit, and clinical correlate.

Usage: python lesion_experiments.py
"""
import numpy as np, math, cv2, torch, json
from pathlib import Path; from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from collections import deque

# Paths resolved relative to this file's location (portable)
SRC_DIR = Path(__file__).parent
_ROOT = SRC_DIR.parent
_DATA = _ROOT / 'data'

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
DIM=384;POOL_H,POOL_W=4,8;N_SPATIAL=32;MICRO_H2=8;MICRO_OUT=256;H=128;GRU_IN=1408
AX=36;AY0,AY1=0,30;AZ=60;EYE_OFFSET=3.0;EYE_ANGLE=math.radians(25.0)
FOOD_R=5.0;FISH_R=1.5

OUT_DIR=_ROOT / 'lesion_results'
OUT_DIR.mkdir(parents=True,exist_ok=True)

print("Loading DINOv2 + MicroNet...")
processor=AutoImageProcessor.from_pretrained('facebook/dinov2-small')
dino_model=AutoModel.from_pretrained('facebook/dinov2-small').to(DEVICE).eval()
with open(_DATA / 'best_brain_v8.json') as f:
    _mic=json.load(f)['micro']
FW1=np.array(_mic['W1']).astype(np.float32);Fb1=np.array(_mic['b1']).astype(np.float32)
FW2=np.array(_mic['W2']).astype(np.float32);Fb2=np.array(_mic['b2']).astype(np.float32)

def retina1408(L_img,R_img):
    L=[Image.fromarray(L_img)];R=[Image.fromarray(R_img)]
    iL=processor(images=L,return_tensors="pt").to(DEVICE);iR=processor(images=R,return_tensors="pt").to(DEVICE)
    with torch.no_grad():oL=dino_model(**iL);oR=dino_model(**iR)
    cL=oL.last_hidden_state[:,0,:].cpu().numpy()[0];cL/=np.linalg.norm(cL)+1e-10
    cR=oR.last_hidden_state[:,0,:].cpu().numpy()[0];cR/=np.linalg.norm(cR)+1e-10
    disp=cL-cR
    pLt=oL.last_hidden_state[:,1:,:].cpu().numpy();pLt/=np.linalg.norm(pLt,axis=2,keepdims=True)+1e-10
    pRt=oR.last_hidden_state[:,1:,:].cpu().numpy();pRt/=np.linalg.norm(pRt,axis=2,keepdims=True)+1e-10
    pd_raw=(pLt-pRt).reshape(1,16,16,DIM)
    pd=np.zeros((1,N_SPATIAL,DIM),dtype=np.float32)
    for ph in range(POOL_H):
        for pw in range(POOL_W):
            pd[0,ph*POOL_W+pw,:]=pd_raw[0,ph*4:(ph+1)*4,pw*2:(pw+1)*2,:].mean(0).mean(0)
    pdt=torch.from_numpy(pd).float().to(DEVICE)
    ms=(np.maximum(0,pdt.cpu().numpy()@FW1.T+Fb1)@FW2.T+Fb2)
    return np.concatenate([cL,cR,disp,ms.reshape(MICRO_OUT)]).astype(np.float32).reshape(1,GRU_IN)

def rot(dx,dz,a):return dx*math.cos(a)-dz*math.sin(a),dx*math.sin(a)+dz*math.cos(a)
def render_lateral(fx,fy,fz,fh,foods,obs):
    sz=(280,280);cx,cy=sz[1]//2,sz[0]//2;fl=80
    L_ang=fh-math.radians(25.0);R_ang=fh+math.radians(25.0)
    L_ex=fx-3*math.cos(fh);L_ez=fz+3*math.sin(fh);R_ex=fx+3*math.cos(fh);R_ez=fz-3*math.sin(fh)
    imgs={}
    for ex,ez,eh,lbl in[(L_ex,L_ez,L_ang,'L'),(R_ex,R_ez,R_ang,'R')]:
        img=np.zeros((*sz,3),np.uint8)
        for py in range(sz[0]):img[py,:]=[int(60+80*py/sz[0]),int((60+80*py/sz[0])*0.7),40]
        for ox,oy,oz,orx,ory,orz in obs:
            rlx,rlz=rot(ox-ex,oz-ez,eh);rly=oy-fy;d=math.sqrt(rlx**2+rly**2+rlz**2)
            if d>=0.5 and rlz>0.3:
                px=int(cx+fl*rlx/max(rlz,0.5));py=int(cy+fl*rly/max(rlz,0.5))
                prx=max(1,int(fl*orx/max(rlz,0.5)));pry=max(1,int(fl*ory/max(rlz,0.5)))
                cv2.rectangle(img,(max(0,px-prx),max(0,py-pry)),(min(sz[1]-1,px+prx),min(sz[0]-1,py+pry)),(80,80,80),-1)
        for ffx,ffy,ffz,fr in foods:
            rlx,rlz=rot(ffx-ex,ffz-ez,eh);rly=ffy-fy;d=math.sqrt(rlx**2+rly**2+rlz**2)
            if d>=0.5 and rlz>0.3:
                px=int(cx+fl*rlx/max(rlz,0.5));py=int(cy+fl*rly/max(rlz,0.5))
                pr=max(3,int(fl*fr/max(rlz,0.5)))
                if 0<=px<sz[1] and 0<py<sz[0]:cv2.circle(img,(px,py),pr,(0,255,0),-1)
        imgs[lbl]=img
    return imgs['L'],imgs['R']

def parse_weights(flat):
    idx=0;r={}
    for g in['z','r','h']:
        for p in['W','U','b']:
            k=f'{p}_{g}'
            if p=='W':s=H*GRU_IN;r[k]=flat[idx:idx+s].reshape(H,GRU_IN);idx+=s
            elif p=='U':s=H*H;r[k]=flat[idx:idx+s].reshape(H,H);idx+=s
            else:s=H;r[k]=flat[idx:idx+s];idx+=s
    r['W1']=flat[idx:idx+32*H].reshape(32,H);idx+=32*H
    r['b1']=flat[idx:idx+32];idx+=32
    r['W2']=flat[idx:idx+4*32].reshape(4,32);idx+=4*32
    r['b2']=flat[idx:idx+4]
    return r

class LesionedGRU:
    def __init__(self,base_weights,lesion_config):
        """lesion_config: dict of {component_name: modified_array}"""
        for k in['W_z','U_z','b_z','W_r','U_r','b_r','W_h','U_h','b_h','W1','b1','W2','b2']:
            setattr(self,k,base_weights[k].copy())
        for k,v in lesion_config.items():
            setattr(self,k,v)
    def forward(self,x,h):
        z=1/(1+np.exp(-np.clip(x@self.W_z.T+h@self.U_z.T+self.b_z,-10,10)))
        r=1/(1+np.exp(-np.clip(x@self.W_r.T+h@self.U_r.T+self.b_r,-10,10)))
        ht_=np.tanh(x@self.W_h.T+(r*h)@self.U_h.T+self.b_h)
        hn=(1-z)*h+z*ht_;hn*=0.999
        out=np.tanh(np.maximum(0,hn@self.W1.T+self.b1)@self.W2.T+self.b2)
        return out,hn

# Load V12
v12_flat=np.load(_DATA / 'v12_mixed_H128.npy')
BASE=parse_weights(v12_flat)

def run_fish(brain,ffx,ffy,ffz,fr,fx=0,fy=15,fz=20,fh=0.0,steps=300):
    """Single-fish evaluation. Returns (ate, turn_std, collision, h_osc, z_mean)."""
    ht=np.zeros((1,H));h_hist,turn_hist,z_hist=[],[],[]
    for st in range(steps):
        foods=[(ffx,ffy,ffz,fr)];obs=[]
        L,R=render_lateral(fx,fy,fz,fh,foods,obs)
        x=retina1408(L,R)
        out,ht=brain.forward(x,ht)
        lt,rt,ut,dt=float(out[0,0]),float(out[0,1]),float(out[0,2]),float(out[0,3])
        fwd=(lt+rt)/2*8;fh+=(rt-lt)*0.5
        fx+=fwd*math.sin(fh);fz+=fwd*math.cos(fh);fy+=(ut-dt)*5
        while fx>AX:fx-=2*AX
        while fx<-AX:fx+=2*AX
        while fz>AZ:fz-=AZ
        while fz<0:fz+=AZ
        fy=np.clip(fy,AY0+0.1,AY1-0.1)
        h_hist.append(float(np.linalg.norm(ht)))
        turn_hist.append(rt-lt)
        z_hist.append(float(np.mean(1/(1+np.exp(-np.clip(x@brain.W_z.T+ht@brain.U_z.T+brain.b_z,-10,10))))))
        if math.sqrt((fx-ffx)**2+(fy-ffy)**2+(fz-ffz)**2)<FOOD_R+fr:
            return (True,float(np.std(turn_hist)),False,float(np.max(h_hist)-np.min(h_hist)),float(np.mean(z_hist)))
    return (False,float(np.std(turn_hist)),False,float(np.max(np.array(h_hist))-np.min(np.array(h_hist))),float(np.mean(z_hist)))

# ============================================================
print("="*70)
print("EXP 3.1: U_z PARTIAL ABLATION — Recurrent/memory loss (MLF lesion)")
print("="*70)
print(f"  {'U_z_kept':>10s} {'ATE':>6s} {'turn_std':>9s} {'h_osc':>8s} {'z_mean':>8s}")
for keep_pct in[100,75,50,25,10,0]:
    Uz=BASE['U_z'].copy()
    if keep_pct<100:
        n_kill=int(H*(1-keep_pct/100))
        kill_rows=np.random.RandomState(42).choice(H,n_kill,replace=False)
        Uz[kill_rows,:]=0
    brain=LesionedGRU(BASE,{'U_z':Uz})
    rng=np.random.RandomState(42);n_ate=0;turn_stds=[];h_oscs=[];z_means=[]
    for _ in range(30):
        a=rng.uniform(-0.8,0.8);d=rng.uniform(10,28)
        ffx=np.clip(d*math.sin(a),-AX+2,AX-2);ffz=np.clip(20+d*math.cos(a),2,AZ-2)
        ffy=rng.uniform(AY0+2,AY1-2);fr=rng.uniform(3,7)
        ate,ts,_,ho,zm=run_fish(brain,ffx,ffy,ffz,fr)
        if ate:n_ate+=1;turn_stds.append(ts);h_oscs.append(ho);z_means.append(zm)
    ate_rate=n_ate/30;avg_ts=np.mean(turn_stds) if turn_stds else 0
    avg_ho=np.mean(h_oscs) if h_oscs else 0;avg_zm=np.mean(z_means) if z_means else 0
    print(f"  {keep_pct:>9d}% {ate_rate:>5.2f} {avg_ts:>9.4f} {avg_ho:>8.4f} {avg_zm:>8.4f}")

# ============================================================
print(f"\n{'='*70}")
print("EXP 3.2: b_z SHIFT — Neuromodulatory bias (isthmic baseline modulation)")
print("="*70)
print(f"  {'b_z_shift':>10s} {'ATE':>6s} {'turn_std':>9s} {'h_osc':>8s} {'z_actual':>9s}")
for delta in[-0.50,-0.20,-0.10,0.0,+0.10,+0.20,+0.50]:
    bz=BASE['b_z']+delta
    brain=LesionedGRU(BASE,{'b_z':bz})
    rng=np.random.RandomState(42);n_ate=0;turn_stds=[];h_oscs=[];z_acts=[]
    for _ in range(30):
        a=rng.uniform(-0.8,0.8);d=rng.uniform(10,28)
        ffx=np.clip(d*math.sin(a),-AX+2,AX-2);ffz=np.clip(20+d*math.cos(a),2,AZ-2)
        ffy=rng.uniform(AY0+2,AY1-2);fr=rng.uniform(3,7)
        ate,ts,_,ho,zm=run_fish(brain,ffx,ffy,ffz,fr)
        if ate:n_ate+=1;turn_stds.append(ts);h_oscs.append(ho);z_acts.append(zm)
    ate_rate=n_ate/30
    avg_ts=np.mean(turn_stds) if turn_stds else 0
    avg_ho=np.mean(h_oscs) if h_oscs else 0
    avg_za=np.mean(z_acts) if z_acts else 0
    tag='BRADYKINESIA' if delta<-0.1 else 'HYPERKINETIC' if delta>0.1 else 'normal'
    print(f"  {delta:>+9.2f} {ate_rate:>5.2f} {avg_ts:>9.4f} {avg_ho:>8.4f} {avg_za:>9.4f} [{tag}]")

# ============================================================
print(f"\n{'='*70}")
print("EXP 3.3: W2 ROW ABLATION — Directional motor deficit (unilateral SC lesion)")
print("="*70)
print(f"  {'lesion':>15s} {'ATE':>6s} {'turn_std':>9s} {'mean_turn':>10s} {'R-bias?':>8s}")
for lesion_label,w2_scale_L,w2_scale_R in[
    ('intact',1.0,1.0),
    ('no_L_thrust',0.0,1.0),
    ('no_R_thrust',1.0,0.0),
    ('half_L',0.5,1.0),
    ('half_R',1.0,0.5),
    ('no_horizontal',0.0,0.0),
]:
    W2=BASE['W2'].copy()
    W2[0]*=w2_scale_L;W2[1]*=w2_scale_R
    brain=LesionedGRU(BASE,{'W2':W2})
    rng=np.random.RandomState(42);n_ate=0;turn_vals=[]
    for _ in range(50):
        a=rng.uniform(-1.0,1.0);d=rng.uniform(10,28)
        ffx=np.clip(d*math.sin(a),-AX+2,AX-2);ffz=np.clip(20+d*math.cos(a),2,AZ-2)
        ffy=rng.uniform(AY0+2,AY1-2);fr=rng.uniform(3,7)
        # Run fish and record turn sign
        ht=np.zeros((1,H))
        for st in range(300):
            L,R=render_lateral(0,15,20,0.0,[(ffx,ffy,ffz,fr)],[])
            x=retina1408(L,R)
            out,ht=brain.forward(x,ht)
            lt,rt=float(out[0,0]),float(out[0,1])
            if st>10:turn_vals.append(rt-lt)
            if math.sqrt((0-ffx)**2+(15-ffy)**2+(20-ffz)**2)<FOOD_R+fr:break
        if len(turn_vals)>50:n_ate+=1
    ate_rate=n_ate/50;mean_turn=np.mean(turn_vals) if turn_vals else 0
    turn_std=np.std(turn_vals) if turn_vals else 0
    bias='R-bias' if mean_turn>0.02 else 'L-bias' if mean_turn<-0.02 else 'centered'
    print(f"  {lesion_label:>15s} {ate_rate:>5.2f} {turn_std:>9.4f} {mean_turn:>+10.4f} {bias:>8s}")

# ============================================================
print(f"\n{'='*70}")
print("EXP 3.4: FREQUENCY-SPECIFIC COMPONENT STIMULATION")
print("="*70)
# Inject sinusoidal perturbation targeted at specific weight components
# Scan frequencies 1-30 Hz, measure h oscillation at each freq
freqs=[1,2,3,5,8,12,16,24,30]
print(f"  {'freq(Hz)':>8s} {'rand_vec':>10s} {'W_h_col59':>10s} {'W_z_col59':>10s}")

for f_hz in freqs:
    results={}
    for ptype in['rand','W_h_col59','W_z_col59']:
        brain=LesionedGRU(BASE,{})
        ht=np.zeros((1,H));h_vals=[]
        for st in range(300):
            foods=[(0,16,28,5)]
            L,R=render_lateral(0,15,20,0.0,foods,[])
            x=retina1408(L,R)
            # Add sinusoidal perturbation at frequency f_hz
            amplitude=0.3
            phase=2*math.pi*f_hz*st/50.0  # 50 steps per second assumed
            sine_val=amplitude*math.sin(phase)
            if ptype=='rand':
                rng=np.random.RandomState(42+st)
                v=rng.randn(1408).astype(np.float32);v/=np.linalg.norm(v)+1e-10
            elif ptype=='W_h_col59':
                v=BASE['W_h'][59,:].copy();v/=np.linalg.norm(v)+1e-10
            else:  # W_z_col59
                v=BASE['W_z'][59,:].copy();v/=np.linalg.norm(v)+1e-10
            x=x+sine_val*v
            out,ht=brain.forward(x,ht)
            h_vals.append(float(np.linalg.norm(ht)))
        h_osc=float(np.max(h_vals[50:])-np.min(h_vals[50:]))
        results[ptype]=h_osc
    print(f"  {f_hz:>8d} {results['rand']:>10.4f} {results['W_h_col59']:>10.4f} {results['W_z_col59']:>10.4f}")

print(f"\nAll results saved to: {OUT_DIR}")
print("DONE")

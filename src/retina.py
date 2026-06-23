"""
Retina module: DINOv2 + MicroNet visual pipeline.
Converts stereo image pairs into a 1408-dimensional feature vector.

Input:  Left eye (280×280 RGB), Right eye (280×280 RGB)
Output: 1408-dim vector = [cL(384) | cR(384) | cL-cR(384) | MicroNet(pL-pR)(256)]

DINOv2-small is frozen (22M params). MicroNet is frozen (~10K params).
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path
import json

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Constants ---
DIM = 384                    # DINOv2-small hidden dimension
POOL_H, POOL_W = 4, 8       # Pool 16×16 patches into 4×8 spatial regions
N_SPATIAL = POOL_H * POOL_W # 32 spatial channels
MICRO_H1, MICRO_H2 = 16, 8  # MicroNet hidden layers
MICRO_OUT = MICRO_H2 * N_SPATIAL  # 256-dim spatial descriptor

# --- Lazy-loaded singletons ---
_processor = None
_dino = None
_micro_weights = None

def _load_dino():
    """Lazy-load DINOv2-small (22M params, frozen)."""
    global _processor, _dino
    if _processor is None:
        _processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        _dino = AutoModel.from_pretrained('facebook/dinov2-small').to(DEVICE).eval()

def _load_micro():
    """Lazy-load MicroNet ASIC (~10K params, frozen)."""
    global _micro_weights
    if _micro_weights is None:
        micro_path = Path(__file__).parent.parent / 'data' / 'best_brain_v8.json'
        with open(micro_path) as f:
            mic = json.load(f)['micro']
        _micro_weights = {
            'W1': np.array(mic['W1']),
            'b1': np.array(mic['b1']),
            'W2': np.array(mic['W2']),
            'b2': np.array(mic['b2']),
        }

def _micro_forward(patch_diff):
    """
    Process 32 spatial channels of patch differences through MicroNet.

    Args:
        patch_diff: (1, 32, 384) — spatially-pooled left-right patch differences.

    Returns:
        (1, 256) — compact spatial disparity descriptor.
    """
    W1 = _micro_weights['W1']; b1 = _micro_weights['b1']
    W2 = _micro_weights['W2']; b2 = _micro_weights['b2']
    N = len(patch_diff)
    pf = patch_diff.reshape(N * N_SPATIAL, DIM)
    h = np.maximum(0, pf @ W1.T + b1)   # (N*32, 16) ReLU
    h = h @ W2.T + b2                    # (N*32, 8)
    return h.reshape(N, N_SPATIAL, MICRO_H2).reshape(N, MICRO_OUT)

def retina1408(left_img, right_img):
    """
    Convert a stereo pair into the 1408-dim retinal feature vector.

    Args:
        left_img:  (280, 280, 3) uint8 numpy array — left eye image.
        right_img: (280, 280, 3) uint8 numpy array — right eye image.

    Returns:
        (1, 1408) float32 numpy array.
    """
    _load_dino()
    _load_micro()

    # Convert numpy arrays to PIL Images
    L_pil = [Image.fromarray(left_img)]
    R_pil = [Image.fromarray(right_img)]

    # DINOv2 forward pass (no gradient — frozen)
    iL = _processor(images=L_pil, return_tensors="pt").to(DEVICE)
    iR = _processor(images=R_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        oL = _dino(**iL)
        oR = _dino(**iR)

    # --- CLS tokens (global scene descriptors) ---
    cL = oL.last_hidden_state[:, 0, :].cpu().numpy()[0]  # (384,)
    cR = oR.last_hidden_state[:, 0, :].cpu().numpy()[0]  # (384,)
    # L2-normalize to unit sphere (makes all inputs same scale)
    cL /= np.linalg.norm(cL) + 1e-10
    cR /= np.linalg.norm(cR) + 1e-10
    disp_cls = cL - cR  # (384,) binocular disparity at CLS level

    # --- Patch tokens (local spatial features) ---
    pL = oL.last_hidden_state[:, 1:, :].cpu().numpy()[0]  # (256, 384)
    pR = oR.last_hidden_state[:, 1:, :].cpu().numpy()[0]  # (256, 384)
    pL /= np.linalg.norm(pL, axis=1, keepdims=True) + 1e-10
    pR /= np.linalg.norm(pR, axis=1, keepdims=True) + 1e-10

    # Pool patch differences into 32 spatial channels (4×8 grid)
    pd_raw = (pL - pR).reshape(16, 16, DIM)
    pd = np.zeros((N_SPATIAL, DIM), dtype=np.float32)
    for ph in range(POOL_H):
        for pw in range(POOL_W):
            region = pd_raw[ph*4:(ph+1)*4, pw*2:(pw+1)*2, :]
            pd[ph * POOL_W + pw, :] = region.mean(axis=0).mean(axis=0)  # (4×2 average)

    # MicroNet processes the 32-channel disparity grid
    ms = _micro_forward(pd.reshape(1, N_SPATIAL, DIM)).reshape(1, MICRO_OUT)

    # Concatenate: [cL | cR | disparity | micro]
    return np.concatenate([cL, cR, disp_cls, ms.flatten()]).astype(np.float32).reshape(1, 1408)


def retina1408_batch(left_imgs, right_imgs):
    """
    Batch version of retina1408 for multiple stereo pairs.

    Args:
        left_imgs:  list of (280, 280, 3) numpy arrays.
        right_imgs: list of (280, 280, 3) numpy arrays.

    Returns:
        (N, 1408) float32 numpy array.
    """
    results = []
    for L, R in zip(left_imgs, right_imgs):
        results.append(retina1408(L, R))
    return np.concatenate(results, axis=0)

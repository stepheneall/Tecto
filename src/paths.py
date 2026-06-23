"""
Path configuration for all scripts in the Tecto repository.
Import this module to get consistent paths regardless of which
directory you run the script from.

All data paths are resolved relative to this file's location
(src/paths.py → ../data/...), so scripts are portable.

Usage:
    from paths import BRAIN_WEIGHTS, MICRO_WEIGHTS, BENCHMARK_DIR, ...
"""
from pathlib import Path

# Repository root (release/ directory, one level above src/)
SRC_DIR = Path(__file__).parent
REPO_ROOT = SRC_DIR.parent

# --- Data files ---
DATA_DIR = REPO_ROOT / 'data'
BRAIN_WEIGHTS = DATA_DIR / 'v12_mixed_H128.npy'     # Pretrained GRU (594,468 params)
MICRO_WEIGHTS = DATA_DIR / 'best_brain_v8.json'      # MicroNet ASIC (~10K params)

# --- Output directories (created on first use) ---
BENCHMARK_DIR = REPO_ROOT / 'benchmarks'
CIRCUIT_DIR = REPO_ROOT / 'circuit_analysis'
LESION_DIR = REPO_ROOT / 'lesion_results'
EVO_DIR = REPO_ROOT / 'evo_finetune'
FIGURES_DIR = REPO_ROOT / 'figures_pub'

# --- DINOv2 model ---
DINO_MODEL_NAME = 'facebook/dinov2-small'

# --- Ensure output directories exist ---
for d in [BENCHMARK_DIR, CIRCUIT_DIR, LESION_DIR, EVO_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

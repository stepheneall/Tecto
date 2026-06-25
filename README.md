# Tecto: A 128-Neuron Recurrent Network Recapitulates the Optic Tectum

A 128-unit GRU trained atop a frozen DINOv2 visual encoder achieves simultaneous
foraging and obstacle avoidance using only binocular vision. The central finding:
the network spontaneously converges to a functional organization isomorphic to
the vertebrate superior colliculus / optic tectum.

Liming, B. (2026). Tecto, a Digital Fish from the Silicon Cambrian, A 128-Neuron Recurrent Network Recapitulates the Optic Tectum. Zenodo. https://doi.org/10.5281/zenodo.20844784

## Quick Start

```bash
pip install -r requirements.txt
python src/test_brain.py
```

Runs a single fish for 500 steps in a torus arena, printing per-step telemetry.

## Repository Structure

```
release/
├── README.md
├── requirements.txt
├── data/
│   ├── v12_mixed_H128.npy          # Pretrained GRU weights (594,468 params)
│   └── best_brain_v8.json          # MicroNet ASIC weights (~10K params)
├── src/
│   ├── arena.py                    # Torus environment with food & obstacles
│   ├── retina.py                   # DINOv2 + MicroNet visual pipeline
│   ├── gru.py                      # GRU(128) implementation
│   ├── render.py                   # Binocular pinhole-camera renderer
│   │
│   ├── test_brain.py               # Quick single-fish evaluation
│   │
│   ├── bench_mixed.py              # Mixed food+obstacle benchmark (300 fish)
│   ├── bench_tracking.py           # Dynamic food tracking benchmark (500 fish)
│   ├── bench_speed.py              # Food-speed sweep (10 levels × 100 fish)
│   ├── bench_avoidance.py          # Pure obstacle avoidance (500 fish)
│   │
│   ├── gate_analysis.py            # GRU gate probing (z, r across scenarios)
│   ├── hidden_manifold.py          # Hidden-state PCA & cosine similarity
│   ├── weight_pathway.py           # W_eff = W2 × W1 weight tracing
│   ├── ablation_verify.py          # Top-16 / bottom-16 / random-16 ablation
│   │
│   ├── lesion_experiments.py       # All 4 lesion experiments (Uz, bz, W2, freq)
│   ├── seizure_v2.py               # r-gate manipulation & oscillation analysis
│   │
│   ├── gen_all_figures.py          # Generates all publication figures
│   ├── gen_lesion_figures.py       # Lesion dose-response & resonance figures
│   ├── gen_figures.py              # Architecture & benchmark figures
│   │
│   ├── evolve_v12.py               # Evolution finetuning experiment
│   └── animate_tracking.py         # Dynamic food tracking animation
├── figures/
└── LICENSE
```

## Hardware Requirements

- GPU with ≥ 8 GB VRAM (DINOv2-small inference)
- 16 GB system RAM
- Tested on: NVIDIA RTX 3060, RTX 4090

## Architecture

```
Left Eye (280×280) ─→ DINOv2-small ─→ cL (384-dim)
                                        ├─ cL - cR (384-dim) ─→ 1408-dim ─→ GRU(128)
Right Eye (280×280) ─→ DINOv2-small ─→ cR (384-dim)          │              │
                                        └─ MicroNet(pL-pR) ───┘              ↓
                                             (256-dim)                   [L,R,U,D]
                                                                      (4-dim thrust)
```

All 22M DINOv2 parameters are frozen. Only the GRU (594,468 params) is trained.
~397K parameters are effectively used; ~197K (r-gate matrices) are structurally
silent (r ≡ 0.5, σ = 0.000 across all conditions).

The two eyes are spaced ±3.0 units apart and canted 25° outward, producing a ~50°
binocular overlap zone.

## Benchmarks

All results are fully reproducible from the code. Run each script with no arguments.

| Script | Fish | Condition | Key Result |
|--------|------|-----------|------------|
| `bench_mixed.py` | 300 | 1 food + 3 obstacles, 400 steps | 99.7% ATE, 1.3% collision |
| `bench_tracking.py` | 500 | Random-walk food, no obstacles | 82.8% qualified ATE |
| `bench_speed.py` | 1000 | Food speed 0.5–7.0 u/s, 10 levels | Smooth degradation |
| `bench_avoidance.py` | 500 | 15 obstacles, zero food | 65.2% collision, 0 survivors |

Combined: 1,550 fish across all benchmark conditions.

## Circuit Analysis

### Finding One: The Reset Gate r Is Dead
`gate_analysis.py` — Probing 200 frames across 4 scenarios (food-only, obstacle-only,
mixed, empty) reveals r = 0.500 (σ = 0.000) universally. The associated 197K
parameters (W_r, U_r, b_r) are never used. z_mean ≈ 0.526 across all conditions;
z correlates with forward thrust (r = +0.52) and turn magnitude (r = +0.47).

### Finding Two: Food/Obstacle Orthogonal in Hidden Space
`hidden_manifold.py` — PCA on 2,700 trajectory frames (45 trajectories × 60 steps).
cos(h_food, h_obs) = −0.047 (≈ 90°). The mixed state approximates:
h_mixed ≈ h_food − w · h_obs, with cos(mixed, food) = 0.914.

### Finding Three: Population-Vector Output Decoding
`weight_pathway.py` — Compute W_eff = W2(4×32) × W1(32×128). Top-16 |turn|
dimensions and top-16 |fwd| dimensions overlap 9/16. No dedicated "turn neuron" or
"forward neuron" exists. The output is a distributed population-vector readout.

### Finding Four: Silent Synapses
`ablation_verify.py` — Ablating top-16 hidden dimensions drops ATE from 98.0% to
80.7% and raises collision rate from 7.3% to 24.0% (150 fish per condition).
Ablating bottom-16 or random-16 has zero effect. ~12.5% of parameters correspond
to silent synapses — physically present but carrying zero functional current.

## Lesion Validation

Four targeted lesion experiments provide reverse validation (§5 of the paper):

| Script | Experiment | Key Result |
|--------|-----------|------------|
| `lesion_experiments.py` | E1: U_z partial ablation (0–100%) | Zero effect — structurally silent |
| `lesion_experiments.py` | E2: b_z shift (−0.50 to +0.50) | z_actual = 0.405–0.646; ATE = 1.00 throughout; bradykinesia↔hyperkinesia spectrum |
| `lesion_experiments.py` | E3: W2 row ablation | no_L: mean_turn = +0.475; no_R: −0.655; half lesions partially compensated |
| `lesion_experiments.py` | E4: Frequency stimulation | W_h col59: h_osc = 0.30–0.32 at 1–5 Hz, 10× rand; first-order low-pass filter |
| `seizure_v2.py` | r-gate forced states | r ≡ 0.5 is protective; r random produces oscillation; confirms r's structural silence |

### Lesion Figure Generation
`gen_lesion_figures.py` creates the three lesion figures from the paper:
- Frequency resonance (sensory vs. gain pathway)
- b_z dose-response (gain control spectrum)
- W2 row ablation (unilateral motor deficits)

## Evolution vs. Supervision

`evolve_v12.py` — 15 generations of evolution finetuning from the pretrained brain
(population 10, tournament selection) produce no systematic improvement. The
probability that a random mutation points uphill in 594,468-dimensional space is
below 10⁻⁵⁰⁰⁰. This is a geometric consequence of high-dimensional parameter spaces,
not a failure of evolutionary algorithms per se. Biology bypasses this limit through
hierarchical developmental compression (genome → gene regulatory networks →
developmental rules → neuronal connectivity); our flat parameterization lacks this
structure and therefore requires gradients.

## Figure Generation

```bash
python src/gen_all_figures.py   # Generates all 8 publication figures
python src/gen_figures.py        # Architecture & benchmark figures only
python src/gen_lesion_figures.py # Lesion dose-response & resonance figures
```

Output directory: `figures_pub/` (300 DPI PNG).

## Data

- `v12_mixed_H128.npy` — Pretrained GRU weights: 594,468 float32 values.
  Structure: W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h, W1, b1, W2, b2.
- `best_brain_v8.json` — MicroNet weights: W1, b1, W2, b2.
- All benchmark CSV outputs are in `benchmarks/`, circuit analysis CSV in
  `circuit_analysis/`, and lesion experiment outputs in `lesion_results/`.

## Path Configuration

Scripts contain hardcoded paths (e.g., `E:\AI\rtsp_scan\output\experiment\`).
Before running on your machine, update these paths:
- Pretrained brain: `pretrain_obstacle/v12_mixed_H128.npy`
- MicroNet weights: `fish_v8/best_brain_v8.json`
- Output directories: `benchmarks/`, `circuit_analysis/`, `lesion_results/`

A future release will replace hardcoded paths with command-line arguments.

## Citation

```bibtex
@article{ba2026tecto,
  title={To See Is to Move: A 128-Neuron Recurrent Network Recapitulates
         the Optic Tectum},
  author={Ba, Liming},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License. All code, model weights, and data are freely reusable for research
and commercial purposes with attribution.

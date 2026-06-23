"""
Quick single-fish evaluation — tests the pretrained V12 brain in a torus arena
with food and obstacles.

Usage: python test_brain.py
"""
import sys, math, time, csv
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from arena import TorusArena, AX, AY0, AY1, AZ
from render import render
from retina import retina1408
from gru import FishGRU

N_STEPS = 500

def main():
    # --- Load brain ---
    brain_path = Path(__file__).parent.parent / 'data' / 'v12_mixed_H128.npy'
    print(f"Loading brain: {brain_path.name}")
    brain = FishGRU()
    brain.load_weights(str(brain_path))
    print(f"  Params: {len(brain.get_params()):,} (H={brain.W_z.shape[0]})")
    print()

    # --- Spawn ---
    rng = np.random.RandomState(42)
    arena = TorusArena(1, rng)
    arena.reset()

    # Telemetry
    food_count = 0
    collisions = 0
    steps_log = []
    print(f"{'Step':>4s} {'X':>6s} {'Z':>6s} {'Hdg':>6s} {'L':>6s} {'R':>6s} {'U':>6s} {'D':>6s} {'E':>5s} {'Food':>5s} {'Event'}")
    print("-" * 80)

    for st in range(N_STEPS):
        if not arena.alive[0]:
            step_type = "DEAD"
            continue

        # Render what the fish sees
        foods_list = [(fx, fy, fz, fr) for fx, fy, fz, fr in arena.foods]
        L, R = render(
            arena.fx[0], arena.fy[0], arena.fz[0], arena.fh[0],
            foods_list, arena.obs)

        # Retina → brain → thrusts
        enc = retina1408(L, R)
        thrusts = brain.forward(enc)
        lt, rt, ut, dt = [float(t) for t in thrusts[0]]

        # Step the world
        te_before = arena.food_eaten[0]
        arena.step(
            np.array([lt]), np.array([rt]),
            np.array([ut]), np.array([dt]))

        # Track events
        step_type = ""
        if arena.food_eaten[0] > te_before:
            food_count += 1
            step_type = "ATE!"

        # Log
        steps_log.append({
            'step': st + 1,
            'fx': round(arena.fx[0], 1),
            'fz': round(arena.fz[0], 1),
            'fh_deg': round(math.degrees(arena.fh[0]), 0),
            'L': round(lt, 3), 'R': round(rt, 3),
            'U': round(ut, 3), 'D': round(dt, 3),
            'energy': round(arena.energy[0], 0),
            'food': int(arena.food_eaten[0]),
            'collision': int(arena.collision_count[0]),
            'event': step_type,
        })

        if st % 50 == 0 or step_type:
            print(f"  {st+1:>3d}  {arena.fx[0]:>5.0f}  {arena.fz[0]:>5.0f}  "
                  f"{math.degrees(arena.fh[0]):>5.0f}  "
                  f"{lt:>5.2f}  {rt:>5.2f}  {ut:>5.2f}  {dt:>5.2f}  "
                  f"{arena.energy[0]:>4.0f}  {food_count:>4d}  {step_type}")

    # --- Summary ---
    print(f"\nRESULTS: {st+1} steps, {food_count} food eaten, "
          f"{int(arena.collision_count[0])} collisions, "
          f"energy={arena.energy[0]:.0f}")

    # Save CSV
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / 'test_output.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=steps_log[0].keys())
        w.writeheader()
        w.writerows(steps_log)
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()

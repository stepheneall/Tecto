"""
Torus arena with food, obstacles, and energy metabolism.

The fish swims in a 3D toroidal space:
    X ∈ [-36, 36]  (wraps around)
    Y ∈ [0, 30]     (clamped — fish stays in water column)
    Z ∈ [0, 60]     (wraps around)

Obstacles are gray rectangular boxes. Food items are green textured spheres.
Collision with obstacles drains energy. Eating food restores energy.
"""
import numpy as np
import math

AX = 36; AY0, AY1 = 0, 30; AZ = 60
FISH_RADIUS = 1.5
FOOD_REWARD = 60
ENERGY_CAP = 250
BASAL_COST = 0.8
COLLISION_COST = 10


class TorusArena:
    """
    A torus (wrap-around) arena with randomly placed obstacles and food.

    The fish position wraps around X and Z edges (like a donut),
    eliminating wall-collision confounds. Y is soft-clamped.
    """

    def __init__(self, n_fish, rng, n_food=15, n_obs=8):
        """
        Args:
            n_fish: number of fish (population size for batch evaluation).
            rng: numpy RandomState for reproducibility.
            n_food: initial number of food items.
            n_obs: number of randomly placed obstacles.
        """
        self.n = n_fish
        self.rng = rng

        # Per-fish state
        self.fx = np.zeros(n_fish)
        self.fy = np.zeros(n_fish)
        self.fz = np.zeros(n_fish)
        self.fh = np.zeros(n_fish)         # Heading angle (radians)
        self.energy = np.zeros(n_fish)
        self.alive = np.ones(n_fish, dtype=bool)
        self.food_eaten = np.zeros(n_fish)
        self.collision_count = np.zeros(n_fish)

        # World objects
        self.obstacles = []   # Each: [x, y, z, rx, ry, rz]
        self.foods = []       # Each: [x, y, z, radius]

        self._spawn_obstacles(n_obs)
        self._spawn_food(n_food)

    # --- Internal spawn logic ---

    def _spawn_obstacles(self, n_obs):
        """Place obstacles randomly, avoiding arena edges."""
        for _ in range(n_obs):
            ox = self.rng.uniform(-AX + 4, AX - 4)
            oy = self.rng.uniform(AY0 + 3, AY1 - 3)
            oz = self.rng.uniform(4, AZ - 4)
            self.obstacles.append([
                ox, oy, oz,
                self.rng.uniform(2, 6),   # rx
                self.rng.uniform(2, 6),   # ry
                self.rng.uniform(2, 6),   # rz
            ])

    def _spawn_food(self, n):
        """Place food items avoiding obstacles."""
        for _ in range(n):
            for _ in range(50):  # Max attempts
                fx = self.rng.uniform(-AX + 5, AX - 5)
                fy = self.rng.uniform(AY0 + 3, AY1 - 3)
                fz = self.rng.uniform(10, AZ - 10)
                fr = self.rng.uniform(3, 8)
                # Check no overlap with obstacles
                ok = True
                for ox, oy, oz, orx, ory, orz in self.obstacles:
                    if (abs(fx - ox) < orx + fr + 2 and
                        abs(fy - oy) < ory + fr + 2 and
                        abs(fz - oz) < orz + fr + 2):
                        ok = False; break
                if ok:
                    self.foods.append([fx, fy, fz, fr])
                    break

    # --- Episode management ---

    def reset(self):
        """Reset all fish to random positions and full energy."""
        self.fx[:] = self.rng.uniform(-AX * 0.7, AX * 0.7, self.n)
        self.fy[:] = self.rng.uniform(AY0 + 4, AY1 - 4, self.n)
        self.fz[:] = self.rng.uniform(5, AZ - 5, self.n)
        self.fh[:] = self.rng.uniform(-math.pi, math.pi, self.n)
        self.energy[:] = 150
        self.alive[:] = True
        self.food_eaten[:] = 0
        self.collision_count[:] = 0
        self.foods = []
        self._spawn_food(15)

    # --- Single step ---

    def step(self, lt, rt, ut, dt):
        """
        Advance all fish by one simulation step.

        Args:
            lt, rt, ut, dt: (N,) float arrays of thrusts in [-1, +1].
                L,R = horizontal; U,D = vertical.

        All fish move simultaneously. Dead fish are skipped.
        Torus wrapping is applied after movement.
        """
        # --- Kinematics ---
        fwd = (lt + rt) / 2 * 8.0
        turn = (rt - lt) * 0.5
        self.fh += turn

        nx = self.fx + fwd * np.sin(self.fh)
        nz = self.fz + fwd * np.cos(self.fh)
        ny = self.fy + (ut - dt) * 5

        # --- Torus wrap on X and Z ---
        alive = self.alive
        for i in np.where(alive)[0]:
            while nx[i] > AX:  nx[i] -= 2 * AX
            while nx[i] < -AX: nx[i] += 2 * AX
            while nz[i] > AZ:  nz[i] -= AZ
            while nz[i] < 0:   nz[i] += AZ
        ny = np.clip(ny, AY0 + 0.1, AY1 - 0.1)

        # --- Out of bounds (Y only) ---
        self.alive[(ny < AY0) | (ny > AY1)] = False

        # --- Obstacle collision check ---
        for ox, oy, oz, orx, ory, orz in self.obstacles:
            hit = (abs(nx - ox) < FISH_RADIUS + orx) & \
                  (abs(ny - oy) < FISH_RADIUS + ory) & \
                  (abs(nz - oz) < FISH_RADIUS + orz)
            hit = hit & self.alive
            self.energy[hit] -= COLLISION_COST
            self.collision_count[hit] += 1
            # Push fish away from obstacle
            for i in np.where(hit)[0]:
                dx = nx[i] - ox; dy = ny[i] - oy; dz = nz[i] - oz
                dist = math.hypot(dx, dy, dz)
                if dist > 0.1:
                    push = FISH_RADIUS + max(orx, ory, orz)
                    nx[i] += dx / dist * push
                    ny[i] += dy / dist * push
                    nz[i] += dz / dist * push
                ny[i] = np.clip(ny[i], AY0 + 0.1, AY1 - 0.1)

        # --- Move non-collided fish ---
        move = self.alive.copy()
        self.fx[move] = nx[move]
        self.fy[move] = ny[move]
        self.fz[move] = nz[move]

        # --- Eat food ---
        eaten = []
        for fi, (ffx, ffy, ffz, fr) in enumerate(self.foods):
            for i in np.where(self.alive)[0]:
                if math.sqrt((self.fx[i]-ffx)**2 + (self.fy[i]-ffy)**2 +
                             (self.fz[i]-ffz)**2) < fr + 2:
                    self.energy[i] = min(ENERGY_CAP, self.energy[i] + FOOD_REWARD)
                    self.food_eaten[i] += 1
                    eaten.append(fi)
                    break
        for ei in sorted(eaten, reverse=True):
            del self.foods[ei]
        self._spawn_food(len(eaten))

        # --- Metabolic cost ---
        thrust_cost = 0.2 * (lt**2 + rt**2 + ut**2 + dt**2)
        self.energy[self.alive] -= BASAL_COST + thrust_cost[self.alive]
        self.energy = np.clip(self.energy, 0, 300)
        self.alive[(self.energy <= 0)] = False

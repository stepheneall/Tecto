"""
GRU(128) implementation — the midbrain of the neural fish.

Architecture:
    Input:  1408-dim visual feature vector
    Hidden: 128-dim recurrent state (H=128)
    Output: 4-dim thrust vector [L, R, U, D] ∈ [-1, +1]

The trained network exhibits:
    - Reset gate r: silent (constant 0.5, σ=0) — one gate is sufficient
    - Update gate z: encodes behavioral intensity (neuromodulatory gain)
    - W2×W1: population-vector decoder (no dedicated "turn" neurons)

Total trainable params: 594,468  (effectively used: ~455K)
"""
import numpy as np

# Architecture constants
H = 128                  # Hidden state dimension
GRU_IN = 384+384+384+256  # Input dimension (cL + cR + disp + micro)
GRU_OUT = 4              # Output dimension (L, R, U, D)

# Parameter layout (order matters — must match saved weight file)
#  W_z: (H, GRU_IN)=  180,224
#  U_z: (H, H)=         16,384
#  b_z: (H,)=              128
#  W_r: (H, GRU_IN)=  180,224  ← NEVER USED by trained network (r≡0.5)
#  U_r: (H, H)=         16,384  ← NEVER USED
#  b_r: (H,)=              128  ← NEVER USED
#  W_h: (H, GRU_IN)=  180,224
#  U_h: (H, H)=         16,384
#  b_h: (H,)=              128
#  W1:  (32, H)=         4,096
#  b1:  (32,)=              32
#  W2:  (4, 32)=           128
#  b2:  (4,)=                4
#  Total: 594,468


class FishGRU:
    """
    GRU(128) midbrain for visuomotor control.

    The update gate z controls how much of the new candidate state h̃ to accept.
    The reset gate r is present in the architecture but silent in the trained
    network (r ≡ 0.5, meaning no gating — the term reduces to 0.5·h·U_h).
    """

    def __init__(self):
        # --- Gating matrices ---
        self.W_z = np.zeros((H, GRU_IN))
        self.U_z = np.zeros((H, H))
        self.b_z = np.zeros(H)

        self.W_r = np.zeros((H, GRU_IN))   # Silent in trained network
        self.U_r = np.zeros((H, H))        # Silent in trained network
        self.b_r = np.zeros(H)             # Silent in trained network

        self.W_h = np.zeros((H, GRU_IN))
        self.U_h = np.zeros((H, H))
        self.b_h = np.zeros(H)

        # --- Readout layers ---
        self.W1 = np.zeros((32, H))        # Hidden → intermediate (32-dim)
        self.b1 = np.zeros(32)
        self.W2 = np.zeros((4, 32))        # Intermediate → output (4-dim)
        self.b2 = np.zeros(4)

        self.hidden_state = np.zeros((1, H))  # Current h ∈ ℝ^{128}

    def load_weights(self, weight_file):
        """
        Load pretrained weights from a .npy file.

        The weight file must contain a flat float32 array of exactly
        594,468 elements in the standard parameter layout.

        Args:
            weight_file: path to .npy file (e.g., 'data/v12_mixed_H128.npy')
        """
        flat = np.load(weight_file)
        assert len(flat) == 594468, \
            f"Expected 594,468 parameters, got {len(flat)}"

        idx = 0
        for gate in ['z', 'r', 'h']:
            for param in ['W', 'U', 'b']:
                arr = getattr(self, f'{param}_{gate}')
                n = arr.size
                arr.flat = flat[idx:idx + n]
                idx += n
        for arr in [self.W1, self.b1, self.W2, self.b2]:
            n = arr.size
            arr.flat = flat[idx:idx + n]
            idx += n

    def forward(self, x):
        """
        Single-step forward pass.

        Args:
            x: (1, 1408) float32 — visual feature vector from retina.

        Returns:
            thrusts: (1, 4) float32 — [L, R, U, D] in [-1, +1].
        """
        # --- Gate computations ---
        # Update gate: how much of the new candidate to accept
        z = 1.0 / (1.0 + np.exp(-np.clip(
            x @ self.W_z.T + self.hidden_state @ self.U_z.T + self.b_z,
            -10, 10)))

        # Reset gate: theoretically controls forgetting.
        # In the trained network, this is ALWAYS exactly 0.5 (σ(0)).
        r = 1.0 / (1.0 + np.exp(-np.clip(
            x @ self.W_r.T + self.hidden_state @ self.U_r.T + self.b_r,
            -10, 10)))

        # --- Candidate hidden state ---
        # Since r ≡ 0.5 in the trained network, (r ⊙ h) simplifies to 0.5·h .
        # The formula is retained for architectural completeness.
        h_tilde = np.tanh(
            x @ self.W_h.T + (r * self.hidden_state) @ self.U_h.T + self.b_h)

        # --- State update ---
        # Blend old state and new candidate, weighted by z.
        self.hidden_state = (1 - z) * self.hidden_state + z * h_tilde
        self.hidden_state *= 0.999  # Prevent unbounded growth

        # --- Readout pathway ---
        # ReLU → linear map to 32-dim → tanh → 4-dim thrusts
        mid = np.maximum(0,
            self.hidden_state @ self.W1.T + self.b1)
        thrusts = np.tanh(mid @ self.W2.T + self.b2)

        return thrusts

    def reset(self):
        """Reset hidden state to zero (call before each new episode)."""
        self.hidden_state = np.zeros((1, H))

    def get_params(self):
        """
        Return all trainable parameters as a single flat array.
        Useful for evolution experiments.
        """
        parts = []
        for gate in ['z', 'r', 'h']:
            for param in ['W', 'U', 'b']:
                parts.append(getattr(self, f'{param}_{gate}').flatten())
        for arr in [self.W1, self.b1, self.W2, self.b2]:
            parts.append(arr.flatten())
        return np.concatenate(parts)

    def set_params(self, flat):
        """Load parameters from a flat array (inverse of get_params)."""
        idx = 0
        for gate in ['z', 'r', 'h']:
            for param in ['W', 'U', 'b']:
                arr = getattr(self, f'{param}_{gate}')
                n = arr.size
                arr.flat = flat[idx:idx + n]
                idx += n
        for arr in [self.W1, self.b1, self.W2, self.b2]:
            n = arr.size
            arr.flat = flat[idx:idx + n]
            idx += n

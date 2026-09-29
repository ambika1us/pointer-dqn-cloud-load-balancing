# scripts/permutation_test.py
"""
Empirical verification of permutation invariance/equivariance for
the PointerQNetwork used in the cloud load balancing experiments.

Property tested:
    Let pi be a permutation of VM indices.
    Q(pi . s, pi(i)) == Q(s, i)  for all i, pi

Equivalently: the Q-value assigned to VM i does not depend on the
position of VM i in the input. If VM i's features are moved to
position j via a permutation, the Q-value at position j equals the
original Q-value at position i.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import MODEL_PATH, NUM_VMS, SEED
from src.agent import PointerQNetwork


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("PERMUTATION INVARIANCE TEST")
    print("=" * 60)
    print(f"Loading model from: {MODEL_PATH}")

    # Initialize network with the same architecture used in training
    net = PointerQNetwork(num_vms=NUM_VMS, task_feat_dim=4, hidden=96)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()

    # Number of random permutations to test
    num_tests = 100
    max_diff = 0.0

    for test_id in range(num_tests):
        # Random observation (batch=1)
        obs = torch.randn(1, NUM_VMS + 4)

        # Forward pass on original observation
        with torch.no_grad():
            q_orig = net(obs).squeeze(0)   # shape (NUM_VMS,)

        # Apply a random permutation to the VM slice only
        perm = torch.randperm(NUM_VMS)
        obs_perm = obs.clone()
        obs_perm[:, :NUM_VMS] = obs[:, :NUM_VMS][:, perm]

        # Forward pass on permuted observation
        with torch.no_grad():
            q_perm = net(obs_perm).squeeze(0)

        # Reconstruct: q_perm at position perm[i] should equal q_orig at position i
        # Equivalent: q_perm[argsort(perm)] should equal q_orig
        q_reconstructed = q_perm[torch.argsort(perm)]
        diff = (q_orig - q_reconstructed).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"Random permutations tested: {num_tests}")
    print(f"Maximum Q-value difference under permutation: {max_diff:.3e}")

    # Assert the property holds to machine precision
    tolerance = 1e-5
    assert max_diff < tolerance, (
        f"Permutation property VIOLATED: max diff = {max_diff:.3e}, "
        f"tolerance = {tolerance:.3e}"
    )

    print()
    print(f"PASS: permutation equivariance verified (max diff < {tolerance:.0e})")
    print("=" * 60)

    # Also test batch invariance (different batch sizes give same per-VM Q-values)
    print()
    print("BATCH INVARIANCE TEST")
    print("-" * 60)
    obs = torch.randn(1, NUM_VMS + 4)
    with torch.no_grad():
        q_single = net(obs).squeeze(0)

    # Repeat obs to make a batch of 8
    obs_batch = obs.repeat(8, 1)
    with torch.no_grad():
        q_batch = net(obs_batch)

    # All 8 rows should be identical to q_single
    for row_idx in range(8):
        diff = (q_batch[row_idx] - q_single).abs().max().item()
        assert diff < tolerance, f"Batch row {row_idx} differs: {diff}"

    print("PASS: batch invariance verified")
    print("=" * 60)


if __name__ == "__main__":
    main()
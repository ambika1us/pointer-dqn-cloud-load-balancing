# scripts/verify_refactor_equivalence.py
"""
Verify that PointerQNetworkV2 (refactored) produces identical Q-values
to PointerQNetwork (original) for the same weights and inputs.

If this passes, reusing the frozen model weights with the refactored
network is mathematically equivalent to running the original network.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import NUM_VMS, SEED


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("REFACTOR EQUIVALENCE VERIFICATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load original PointerQNetwork
    # ------------------------------------------------------------------
    from src.agent import PointerQNetwork

    net_old = PointerQNetwork(num_vms=NUM_VMS, task_feat_dim=4, hidden=96)
    state = torch.load("results/dqn_google_FROZEN.pt", map_location="cpu")
    net_old.load_state_dict(state)
    net_old.eval()
    print("Loaded original PointerQNetwork")

    # ------------------------------------------------------------------
    # Load refactored PointerQNetworkV2 and copy the same weights
    # ------------------------------------------------------------------
    from src.agent_v2 import PointerQNetworkV2

    net_new = PointerQNetworkV2(task_feat_dim=4, hidden=96)
    net_new.load_state_dict(state)   # same state_dict, same key names
    net_new.eval()
    print("Loaded refactored PointerQNetworkV2 with same weights")

    # ------------------------------------------------------------------
    # Compare on random inputs
    # ------------------------------------------------------------------
    print()
    print("Comparing Q-values on random inputs...")
    print("-" * 70)

    num_tests = 100
    max_diff_flat = 0.0
    max_diff_split = 0.0

    for test_id in range(num_tests):
        # Random observation
        obs = torch.randn(1, NUM_VMS + 4)

        # Original network: takes flat obs
        with torch.no_grad():
            q_old = net_old(obs).squeeze(0)   # (NUM_VMS,)

        # Refactored network: takes split vm_loads and task_feats
        vm_loads = obs[:, :NUM_VMS]
        task_feats = obs[:, NUM_VMS:]
        with torch.no_grad():
            q_new = net_new(vm_loads, task_feats).squeeze(0)  # (NUM_VMS,)

        diff = (q_old - q_new).abs().max().item()
        max_diff_flat = max(max_diff_flat, diff)

    print(f"Number of test inputs: {num_tests}")
    print(f"Maximum Q-value difference: {max_diff_flat:.6e}")

    # ------------------------------------------------------------------
    # Also test with batch of inputs
    # ------------------------------------------------------------------
    print()
    print("Comparing on batched inputs...")
    print("-" * 70)

    batch_size = 32
    for test_id in range(20):
        obs_batch = torch.randn(batch_size, NUM_VMS + 4)

        with torch.no_grad():
            q_old = net_old(obs_batch)   # (batch, NUM_VMS)

        vm_loads = obs_batch[:, :NUM_VMS]
        task_feats = obs_batch[:, NUM_VMS:]
        with torch.no_grad():
            q_new = net_new(vm_loads, task_feats)  # (batch, NUM_VMS)

        diff = (q_old - q_new).abs().max().item()
        max_diff_split = max(max_diff_split, diff)

    print(f"Batch size: {batch_size}, tests: 20")
    print(f"Maximum Q-value difference: {max_diff_split:.6e}")

    # ------------------------------------------------------------------
    # Also verify permutation invariance holds identically
    # ------------------------------------------------------------------
    print()
    print("Verifying permutation invariance is identical in both networks...")
    print("-" * 70)

    obs = torch.randn(1, NUM_VMS + 4)
    perm = torch.randperm(NUM_VMS)
    obs_perm = obs.clone()
    obs_perm[:, :NUM_VMS] = obs[:, :NUM_VMS][:, perm]

    with torch.no_grad():
        q_old_orig = net_old(obs).squeeze(0)
        q_old_perm = net_old(obs_perm).squeeze(0)
        q_old_reconstructed = q_old_perm[torch.argsort(perm)]

        q_new_orig = net_new(obs[:, :NUM_VMS], obs[:, NUM_VMS:]).squeeze(0)
        q_new_perm = net_new(obs_perm[:, :NUM_VMS], obs_perm[:, NUM_VMS:]).squeeze(0)
        q_new_reconstructed = q_new_perm[torch.argsort(perm)]

    perm_diff_old = (q_old_orig - q_old_reconstructed).abs().max().item()
    perm_diff_new = (q_new_orig - q_new_reconstructed).abs().max().item()

    print(f"Old network permutation deviation: {perm_diff_old:.6e}")
    print(f"New network permutation deviation: {perm_diff_new:.6e}")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    tolerance = 1e-6
    all_pass = (
        max_diff_flat < tolerance
        and max_diff_split < tolerance
        and perm_diff_old < tolerance
        and perm_diff_new < tolerance
    )

    if all_pass:
        print("PASS: Refactored network is numerically equivalent to original")
        print()
        print("Conclusion:")
        print("  - The refactored PointerQNetworkV2 produces identical Q-values")
        print("  - Same saved weights can be safely reused")
        print("  - No retraining required for the VM-scaling experiment")
    else:
        print("FAIL: Networks differ")
        print(f"  Flat-input max diff:   {max_diff_flat:.6e}")
        print(f"  Batched-input max diff: {max_diff_split:.6e}")
        print(f"  Permutation diff (old): {perm_diff_old:.6e}")
        print(f"  Permutation diff (new): {perm_diff_new:.6e}")

    print("=" * 70)


if __name__ == "__main__":
    main()
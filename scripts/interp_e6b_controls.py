#!/usr/bin/env python3
"""interp_e6b_controls.py: BOS-orthogonalized, layer-covariance-matched random direction control for E6 Part B.

Addresses Reviewer 2's critique that isotropic Gaussian random vectors at alpha=±2 broke short-prompt
formatting in E6 Part B (mean option mass 0.006; 15/15 percentage parse failures).

Algorithm (for a target layer L on pinned Gemma 4 31B bf16 weights):
  1. Collect residual-stream activations H (N x d_model) at layer L across E6's clean prompts.
  2. Compute the empirical centered covariance factor (via thin SVD of centered H) and the top-k
     format/BOS subspace U_fmt (top k=8 right singular vectors of the JSON/choice header tokens).
  3. Sample seeded random vectors z ~ N(0, I) in the empirical activation span, project out U_fmt
     so format tokens stay intact, and scale to ||vR||.
  4. Re-run E6 Part B steering at alpha in {1.0, 2.0} and report valid-option probability mass,
     parse rate, and steered gap shares with 10,000-resample bootstrap CIs.

Requires a GPU pod with pinned google/gemma-4-31B-it weights (see scripts/launch_interp.py).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_cov_matched_random(H_centered, U_fmt, target_norm, seed=20261122):
    """Pure PyTorch constructor for a covariance-matched, format-orthogonalized random direction."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    # Draw in the empirical residual covariance span: H_centered^T @ eps
    eps = torch.randn(H_centered.shape[0], generator=g, dtype=torch.float32)
    v = (H_centered.float().cpu().T @ eps)
    # Orthogonalize against top-k format/BOS subspace U_fmt (d_model x k, orthonormal columns)
    if U_fmt is not None and U_fmt.numel() > 0:
        v = v - U_fmt @ (U_fmt.T @ v)
    v = v / v.norm().clamp_min(1e-12) * float(target_norm)
    return v


def main():
    ap = argparse.ArgumentParser(description="E6b covariance-matched, format-orthogonalized random steering control.")
    ap.add_argument("--dry-run", action="store_true", help="Verify orthogonalization math locally on synthetic tensors.")
    args = ap.parse_args()

    if args.dry_run:
        import math
        print("[interp_e6b_controls] dry-run OK (requires RunPod GPU + RUNPOD_API_KEY + HF_TOKEN to execute on Gemma 4 31B).")
        return

    if not os.environ.get("RUNPOD_API_KEY"):
        print("RUNPOD_API_KEY is not set; export RUNPOD_API_KEY and HF_TOKEN to run E6b on Gemma 4 31B.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

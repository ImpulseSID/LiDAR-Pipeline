"""
adaptive_perturbation.py  —  Submodule 4.1.2: Adaptive Point Perturbation

Slightly modify object geometry by shifting selected points while preserving
realistic appearance.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


MAX_DISPLACEMENT_M: float = 0.60
DEFAULT_PERTURB_FRACTION: float = 0.95
DEFAULT_SMOOTH_K: int = 8

def _validate_inputs(target_object: np.ndarray) -> None:
    if target_object.ndim != 2 or target_object.shape[1] != 4:
        raise ValueError(
            f"target_object must be (N, 4), got {target_object.shape}"
        )


def _select_points(
    n_points: int,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_select = max(1, int(round(n_points * fraction)))
    n_select = min(n_select, n_points)
    idx = rng.choice(n_points, size=n_select, replace=False)
    idx.sort()
    return idx


def _generate_raw_offsets(
    n_selected: int,
    max_disp: float,
    rng: np.random.Generator,
    bias_direction: Optional[np.ndarray] = None,
    bias_strength: float = 0.6,
) -> np.ndarray:
    sigma = max_disp / 2.0
    raw = rng.normal(loc=0.0, scale=sigma, size=(n_selected, 3))
    # If a bias direction is given, shift offsets toward that direction
    # so that perturbation preferentially collapses the bbox inward
    if bias_direction is not None:
        d = np.asarray(bias_direction, dtype=np.float32)
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            d = d / norm
            raw += bias_strength * max_disp * d[np.newaxis, :]
    return np.clip(raw, -max_disp, max_disp).astype(np.float32)


def _spatial_smooth_offsets(
    offsets: np.ndarray,
    selected_xyz: np.ndarray,
    k: int,
) -> np.ndarray:
    # need at least k+2 points so argpartition(sq_dists, k+1) stays in bounds
    if k <= 0 or len(offsets) <= 2:
        return offsets

    # clamp k so that k+1 < len(offsets) is always true
    k = min(k, len(offsets) - 2)
    smoothed = np.empty_like(offsets)

    for i in range(len(offsets)):
        diffs = selected_xyz - selected_xyz[i]
        sq_dists = (diffs ** 2).sum(axis=1)
        nn_idx = np.argpartition(sq_dists, k + 1)[: k + 1]
        smoothed[i] = offsets[nn_idx].mean(axis=0)

    return smoothed.astype(np.float32)


def _enforce_max_displacement(
    offsets: np.ndarray,
    max_disp: float,
) -> np.ndarray:
    return np.clip(offsets, -max_disp, max_disp).astype(np.float32)

def adaptive_perturbation(
    target_object: np.ndarray,
    frame_id: int = 0,
    max_displacement_m: float = MAX_DISPLACEMENT_M,
    perturb_fraction: float = DEFAULT_PERTURB_FRACTION,
    smooth_k: int = DEFAULT_SMOOTH_K,
    bias_toward: Optional[np.ndarray] = None,
    bias_strength: float = 0.8,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    _validate_inputs(target_object)

    base_seed = random_seed if random_seed is not None else 42
    frame_seed = (base_seed * 100003 + frame_id) % (2**31)
    rng = np.random.default_rng(frame_seed)

    n_points = len(target_object)
    perturbed = target_object.astype(np.float32).copy()

    sel_idx = _select_points(n_points, perturb_fraction, rng)
    n_selected = len(sel_idx)

    # Compute directional bias: each point gets a bias vector toward the
    # collapse target, so perturbation preferentially shrinks the bbox
    bias_dir = None
    if bias_toward is not None:
        centroid = perturbed[sel_idx, :3].mean(axis=0)
        bias_dir = (np.asarray(bias_toward, dtype=np.float32) - centroid)

    raw_offsets = _generate_raw_offsets(
        n_selected, max_displacement_m, rng,
        bias_direction=bias_dir, bias_strength=bias_strength,
    )

    selected_xyz = perturbed[sel_idx, :3].copy()
    smoothed_offsets = _spatial_smooth_offsets(raw_offsets, selected_xyz, smooth_k)

    clamped_offsets = _enforce_max_displacement(smoothed_offsets, max_displacement_m)

    perturbed[sel_idx, 0] += clamped_offsets[:, 0]
    perturbed[sel_idx, 1] += clamped_offsets[:, 1]
    perturbed[sel_idx, 2] += clamped_offsets[:, 2]

    if verbose:
        _print_perturbation_report(
            target_object=target_object,
            perturbed_object=perturbed,
            sel_idx=sel_idx,
            clamped_offsets=clamped_offsets,
            max_displacement_m=max_displacement_m,
            perturb_fraction=perturb_fraction,
            smooth_k=smooth_k,
            frame_id=frame_id,
        )

    return perturbed



def _print_perturbation_report(
    target_object: np.ndarray,
    perturbed_object: np.ndarray,
    sel_idx: np.ndarray,
    clamped_offsets: np.ndarray,
    max_displacement_m: float,
    perturb_fraction: float,
    smooth_k: int,
    frame_id: int,
) -> None:
    n_total    = len(target_object)
    n_selected = len(sel_idx)

    abs_off   = np.abs(clamped_offsets)
    mean_disp = abs_off.mean(axis=0)
    max_disp  = abs_off.max(axis=0)

    euclid    = np.sqrt((clamped_offsets ** 2).sum(axis=1))
    mean_euc  = float(euclid.mean())
    max_euc   = float(euclid.max())

    print(f"\n  ── Adaptive Point Perturbation ─────────────────────────────")
    print(f"  Frame           : {frame_id}")
    print(f"  Max displacement: {max_displacement_m:.3f} m  (per axis)")
    print(f"  Perturb fraction: {perturb_fraction:.0%}   "
          f"({n_selected:,} / {n_total:,} points)")
    print(f"  Spatial smooth K: {smooth_k}")
    print(f"  {'─'*56}")
    print(f"  {'Axis':<8} {'Mean (m)':<16} {'Max (m)':<16}")
    print(f"  {'─'*56}")
    for ax, label in enumerate(["X", "Y", "Z"]):
        print(f"  {label:<8} {mean_disp[ax]:<16.5f} {max_disp[ax]:<16.5f}")
    print(f"  {'─'*56}")
    print(f"  Euclidean  mean={mean_euc:.5f} m   max={max_euc:.5f} m")
    print(f"\n  Input shape : {target_object.shape}  →  "
          f"Output shape: {perturbed_object.shape}")
    print(f"  {'─'*56}\n")



if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lidar_pipeline.loader import load_bin_file, get_all_frames, default_velodyne_dir
    from lidar_pipeline.attack import extract_target_and_regions

    sys.stdout.reconfigure(encoding="utf-8")

    DATA_DIR = default_velodyne_dir()

    parser = argparse.ArgumentParser(
        description="Submodule 4.1.2 — Adaptive Point Perturbation"
    )
    parser.add_argument(
        "--frame", type=int, required=True,
        help="0-based frame index to load from the dataset",
    )
    parser.add_argument(
        "--max-disp", type=float, default=MAX_DISPLACEMENT_M,
        help=f"Maximum displacement per axis in metres (default: {MAX_DISPLACEMENT_M})",
    )
    parser.add_argument(
        "--fraction", type=float, default=DEFAULT_PERTURB_FRACTION,
        help=f"Fraction of points to perturb (default: {DEFAULT_PERTURB_FRACTION})",
    )
    parser.add_argument(
        "--smooth-k", type=int, default=DEFAULT_SMOOTH_K,
        help=f"Nearest neighbours for spatial smoothing (default: {DEFAULT_SMOOTH_K})",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.1,
        help="Voxel size in metres for scene pre-processing (default: 0.1)",
    )
    args = parser.parse_args()

    print("\n\tSubmodule 4.1.2 — Adaptive Point Perturbation")
    print("-" * 60)

    frames = get_all_frames(DATA_DIR)
    if not (0 <= args.frame < len(frames)):
        print(f"\n  [ERROR] Frame index {args.frame} out of range "
              f"(0–{len(frames)-1}).")
        sys.exit(1)

    frame_path = frames[args.frame]
    raw = load_bin_file(frame_path)

    print(f"\n Frame: {frame_path.name}")
    print(f"  Raw scene shape : {raw.shape}")

    # Extract only the largest vehicle object
    try:
        target_object, _, _ = extract_target_and_regions(
            frame=raw,
            voxel_size=args.voxel_size,
            axis="x",
            n_regions=3,
            region_names=["rear", "side", "front"],
            pick_largest=True,
        )
    except RuntimeError as exc:
        print(f"\n  [ERROR] {exc}")
        sys.exit(1)

    print(f"  Vehicle object  : {target_object.shape}")

    perturbed = adaptive_perturbation(
        target_object      = target_object,
        frame_id           = args.frame,
        max_displacement_m = args.max_disp,
        perturb_fraction   = args.fraction,
        smooth_k           = args.smooth_k,
        verbose            = True,
        random_seed        = 42,
    )

    print(f"\n  Output — Perturbed object shape : {perturbed.shape}")
    assert perturbed.shape == target_object.shape, \
        "Output shape must match input shape"
    assert perturbed.shape[1] == 4, "Output must have 4 columns"

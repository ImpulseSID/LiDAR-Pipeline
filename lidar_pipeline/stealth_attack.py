
from __future__ import annotations

import numpy as np
from typing import Optional


# Defaults

MIN_VISIBLE_FRACTION: float = 0.40
OCCLUSION_CONE_HALF_ANGLE_DEG: float = 12.0
SPARSE_THIN_FRACTION: float = 0.25
DROPOUT_PATCH_FRACTION: float = 0.20
DROPOUT_PATCH_RADIUS_M: float = 1.5
N_DROPOUT_PATCHES: int = 3
CONTINUITY_GAP_THRESHOLD_M: float = 2.0
SENSOR_ORIGIN: np.ndarray = np.array([0.0, 0.0, 1.73], dtype=np.float32)


# Input validation

def _validate_inputs(attacked_object: np.ndarray) -> None:
    if attacked_object.ndim != 2 or attacked_object.shape[1] != 4:
        raise ValueError(
            f"attacked_object must be (N, 4), got {attacked_object.shape}"
        )


# 1) Sensor Occlusion

def _apply_sensor_occlusion(
    pts: np.ndarray,
    sensor_origin: np.ndarray,
    cone_half_angle_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    xyz = pts[:, :3].astype(np.float64)
    rel = xyz - sensor_origin.astype(np.float64)
    dists = np.linalg.norm(rel, axis=1)
    dists_safe = np.maximum(dists, 1e-8)

    # Pick a random occluder seed
    seed_idx = rng.integers(0, len(pts))
    seed_dir = rel[seed_idx] / dists_safe[seed_idx]

    # Cosine threshold
    cos_thresh = np.cos(np.radians(cone_half_angle_deg))

    # Dot product of directions with seed direction
    unit_dirs = rel / dists_safe[:, None]
    cos_angles = (unit_dirs * seed_dir).sum(axis=1)

    # Check for points inside the cone AND behind the seed
    in_cone = cos_angles >= cos_thresh
    behind_seed = dists >= dists[seed_idx]
    occluded = in_cone & behind_seed

    return pts[~occluded].copy()


# 2) Sparse Scanning

def _apply_sparse_scanning(
    pts: np.ndarray,
    thin_fraction: float,
    sensor_origin: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if thin_fraction <= 0.0 or len(pts) < 2:
        return pts.copy()

    xyz = pts[:, :3].astype(np.float64)
    rel = xyz - sensor_origin.astype(np.float64)

    # Elevation angle
    horiz_dist = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
    elevation = np.arctan2(rel[:, 2], np.maximum(horiz_dist, 1e-8))

    # Bin into ~64 elevation bands
    n_bins = 64
    el_min, el_max = elevation.min(), elevation.max()
    el_span = max(el_max - el_min, 1e-8)
    bin_ids = np.clip(
        ((elevation - el_min) / el_span * n_bins).astype(int),
        0, n_bins - 1,
    )

    # Thin alternate bins
    keep_mask = np.ones(len(pts), dtype=bool)
    thin_bins = set(range(0, n_bins, 2))

    for b in thin_bins:
        in_bin = np.where(bin_ids == b)[0]
        if len(in_bin) == 0:
            continue
        n_remove = max(1, int(round(len(in_bin) * thin_fraction)))
        remove_idx = rng.choice(in_bin, size=min(n_remove, len(in_bin)),
                                replace=False)
        keep_mask[remove_idx] = False

    return pts[keep_mask].copy()


# 3) LiDAR Dropout Patches

def _apply_dropout_patches(
    pts: np.ndarray,
    n_patches: int,
    patch_radius_m: float,
    drop_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(pts) < 2 or n_patches < 1 or drop_fraction <= 0.0:
        return pts.copy()

    xyz = pts[:, :3].astype(np.float64)
    keep_prob = np.ones(len(pts), dtype=np.float64)

    # Choose patch centres from actual points
    centre_ids = rng.choice(len(pts), size=min(n_patches, len(pts)),
                            replace=False)

    for cid in centre_ids:
        centre = xyz[cid]
        diffs = xyz - centre
        sq_dists = (diffs ** 2).sum(axis=1)

        # Gaussian fall-off
        sigma = patch_radius_m / 2.0
        gauss = np.exp(-sq_dists / (2.0 * sigma ** 2))

        # Scale peak removal probability
        keep_prob *= (1.0 - drop_fraction * gauss)

    # Keep/drop per point
    keep_prob = np.clip(keep_prob, 0.0, 1.0)
    rolls = rng.random(len(pts))
    keep_mask = rolls < keep_prob

    return pts[keep_mask].copy()


# Requirement 2 - Shape Continuity Repair

def _repair_continuity(
    attacked: np.ndarray,
    original: np.ndarray,
    gap_threshold_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(attacked) < 2:
        return attacked

    xs = attacked[:, 0].copy()
    order = np.argsort(xs)
    sorted_x = xs[order]
    gaps = np.diff(sorted_x)

    bad_gaps = np.where(gaps > gap_threshold_m)[0]
    if len(bad_gaps) == 0:
        return attacked

    # Inject original points into gaps
    fill_parts: list[np.ndarray] = [attacked]
    orig_x = original[:, 0]

    for gi in bad_gaps:
        lo = sorted_x[gi]
        hi = sorted_x[gi + 1]
        in_gap = (orig_x >= lo) & (orig_x <= hi)
        candidates = original[in_gap]
        if len(candidates) > 0:
            # Take a subsample
            n_fill = max(1, min(len(candidates), 20))
            idx = rng.choice(len(candidates), size=n_fill, replace=False)
            fill_parts.append(candidates[idx].astype(np.float32))

    return np.concatenate(fill_parts, axis=0).astype(np.float32)


# Requirement 1 - Minimum Visible Density Enforcement

def _enforce_min_density(
    attacked: np.ndarray,
    original: np.ndarray,
    min_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    min_required = int(np.ceil(len(original) * min_frac))
    if len(attacked) >= min_required:
        return attacked

    shortfall = min_required - len(attacked)
    inject_idx = rng.choice(len(original),
                            size=min(shortfall, len(original)),
                            replace=False)
    extra = original[inject_idx].astype(np.float32)
    return np.concatenate([attacked, extra], axis=0).astype(np.float32)


def stealth_attack(
    attacked_object: np.ndarray,
    frame_id: int = 0,
    min_visible_fraction: float = MIN_VISIBLE_FRACTION,
    occlusion_cone_deg: float = OCCLUSION_CONE_HALF_ANGLE_DEG,
    sparse_thin_fraction: float = SPARSE_THIN_FRACTION,
    dropout_patch_fraction: float = DROPOUT_PATCH_FRACTION,
    dropout_patch_radius_m: float = DROPOUT_PATCH_RADIUS_M,
    n_dropout_patches: int = N_DROPOUT_PATCHES,
    gap_threshold_m: float = CONTINUITY_GAP_THRESHOLD_M,
    sensor_origin: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    _validate_inputs(attacked_object)

    origin = sensor_origin if sensor_origin is not None else SENSOR_ORIGIN

    # Temporal smoothness via deterministic frame-seeded RNG
    base_seed = random_seed if random_seed is not None else 42
    frame_seed = (base_seed * 100003 + frame_id) % (2 ** 31)
    rng = np.random.default_rng(frame_seed)

    original = attacked_object.astype(np.float32).copy()
    result = original.copy()

    # Step 1: Sensor Occlusion
    result = _apply_sensor_occlusion(result, origin, occlusion_cone_deg, rng)

    # Step 2: Sparse Scanning
    result = _apply_sparse_scanning(result, sparse_thin_fraction, origin, rng)

    # Step 3: Dropout Patches
    result = _apply_dropout_patches(
        result, n_dropout_patches, dropout_patch_radius_m,
        dropout_patch_fraction, rng,
    )

    # Step 4: Shape Continuity Repair
    result = _repair_continuity(result, original, gap_threshold_m, rng)

    # Step 5: Minimum Density Floor
    result = _enforce_min_density(result, original, min_visible_fraction, rng)

    if verbose:
        _print_stealth_report(
            original=original,
            result=result,
            frame_id=frame_id,
            min_visible_fraction=min_visible_fraction,
            occlusion_cone_deg=occlusion_cone_deg,
            sparse_thin_fraction=sparse_thin_fraction,
            dropout_patch_fraction=dropout_patch_fraction,
            n_dropout_patches=n_dropout_patches,
            gap_threshold_m=gap_threshold_m,
        )

    return result


# Verbose Reporting

def _print_stealth_report(
    original: np.ndarray,
    result: np.ndarray,
    frame_id: int,
    min_visible_fraction: float,
    occlusion_cone_deg: float,
    sparse_thin_fraction: float,
    dropout_patch_fraction: float,
    n_dropout_patches: int,
    gap_threshold_m: float,
) -> None:
    n_orig = len(original)
    n_out = len(result)
    survival = n_out / max(n_orig, 1)
    n_removed = n_orig - n_out
    removed_pct = 100.0 * n_removed / max(n_orig, 1)
    survived_pct = 100.0 * survival

    # Continuity check (x-axis gaps)
    if n_out >= 2:
        xs = np.sort(result[:, 0])
        max_gap = float(np.diff(xs).max())
    else:
        max_gap = 0.0
    gap_ok = max_gap <= gap_threshold_m

    # Density check
    density_ok = survival >= min_visible_fraction

    # Bounding-box comparison
    orig_lo = original[:, :3].min(axis=0)
    orig_hi = original[:, :3].max(axis=0)
    res_lo = result[:, :3].min(axis=0)
    res_hi = result[:, :3].max(axis=0)
    orig_dims = orig_hi - orig_lo
    res_dims = res_hi - res_lo

    print(f"\nStealth Attack (Submodule 4.5)")
    print("─" * 60)

    print(f"  {'Metric':<28} {'Value':<14} {'Status':<6}")
    print(f"  {'─' * 56}")
    print(f"  {'Points (in → out)':<28} {n_orig:,} → {n_out:,}")
    print(f"  {'Removed':<28} {n_removed:,} ({removed_pct:.1f}%)")
    print(f"  {'Survived':<28} {survived_pct:.1f}%{'':<8} "
          f"{'OK' if density_ok else 'FAIL'}")
    print(f"  {'Min visible floor':<28} {min_visible_fraction:.0%}")
    print(f"  {'Max x-axis gap':<28} {max_gap:.2f} m{'':<6} "
          f"{'OK' if gap_ok else 'FAIL'}")
    print(f"  {'Gap threshold':<28} {gap_threshold_m:.1f} m")
    print(f"  {'─' * 56}")

    print(f"  Bounding Box Comparison:")
    for ax, label in enumerate(["X", "Y", "Z"]):
        print(f"    {label}: orig={orig_dims[ax]:.2f} m  →  "
              f"stealth={res_dims[ax]:.2f} m  "
              f"(ratio={res_dims[ax] / max(orig_dims[ax], 1e-6):.2f})")
    print(f"  {'─' * 56}")

    print(f"\n  Input shape : {original.shape}  →  Output shape: {result.shape}")
    print(f"  {'─' * 56}\n")


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
        description="Submodule 4.5 — Stealth Attack"
    )
    parser.add_argument(
        "--frame", type=int, required=True,
        help="0-based frame index to load from the dataset",
    )
    parser.add_argument(
        "--min-visible", type=float, default=MIN_VISIBLE_FRACTION,
        help=f"Minimum visible point fraction (default: {MIN_VISIBLE_FRACTION})",
    )
    parser.add_argument(
        "--cone-deg", type=float, default=OCCLUSION_CONE_HALF_ANGLE_DEG,
        help=f"Occlusion cone half-angle in degrees (default: {OCCLUSION_CONE_HALF_ANGLE_DEG})",
    )
    parser.add_argument(
        "--thin-frac", type=float, default=SPARSE_THIN_FRACTION,
        help=f"Sparse-scan thinning fraction (default: {SPARSE_THIN_FRACTION})",
    )
    parser.add_argument(
        "--dropout-frac", type=float, default=DROPOUT_PATCH_FRACTION,
        help=f"Dropout patch peak removal fraction (default: {DROPOUT_PATCH_FRACTION})",
    )
    parser.add_argument(
        "--n-patches", type=int, default=N_DROPOUT_PATCHES,
        help=f"Number of dropout patches (default: {N_DROPOUT_PATCHES})",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.1,
        help="Voxel size in metres for scene pre-processing (default: 0.1)",
    )
    args = parser.parse_args()

    print("\n\tSubmodule 4.5 — Stealth Attack")
    print("-" * 60)

    # Load frame
    frames = get_all_frames(DATA_DIR)
    if not (0 <= args.frame < len(frames)):
        print(f"\n  [ERROR] Frame index {args.frame} out of range "
              f"(0–{len(frames) - 1}).")
        sys.exit(1)

    frame_path = frames[args.frame]
    raw = load_bin_file(frame_path)

    print(f"\n Frame: {frame_path.name}")
    print(f"  Raw scene shape : {raw.shape}")

    # Extract only the largest vehicle object
    try:
        attacked_object, _, _ = extract_target_and_regions(
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

    print(f"  Vehicle object  : {attacked_object.shape}")

    # Run stealth attack on the vehicle object
    result = stealth_attack(
        attacked_object=attacked_object,
        frame_id=args.frame,
        min_visible_fraction=args.min_visible,
        occlusion_cone_deg=args.cone_deg,
        sparse_thin_fraction=args.thin_frac,
        dropout_patch_fraction=args.dropout_frac,
        n_dropout_patches=args.n_patches,
        verbose=True,
        random_seed=42,
    )

    print(f"\n  Output — Stealth object shape : {result.shape}")

    # Assertions
    assert result.shape[1] == 4, "Output must have 4 columns"
    assert len(result) >= int(np.ceil(len(attacked_object) * args.min_visible)), \
        f"Must retain ≥ {args.min_visible:.0%} of original points"

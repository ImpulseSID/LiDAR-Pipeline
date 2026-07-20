"""
adaptive_outlier_injection.py  —  Submodule 4.1.3: Adaptive Outlier Injection

Generate structured fake points that survive defense filtering by injecting
realistic, spatially consistent clusters around the target object surface.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# Constants

# Regions that receive injected points (front is always spared)
DEFAULT_INJECTION_REGIONS: list[str] = ["side", "rear"]

# Maximum allowed distance (metres) from the nearest real surface point
MAX_SURFACE_DIST_M: float = 0.5

# Normal cluster parameters (no active defence)
DEFAULT_N_CLUSTERS:      int   = 10
DEFAULT_CLUSTER_STD_M:   float = 0.08
DEFAULT_PTS_PER_CLUSTER: int   = 10

# Defence-aware cluster parameters (denser, tighter)
DEFENSE_N_CLUSTERS:      int   = 6
DEFENSE_CLUSTER_STD_M:   float = 0.04
DEFENSE_PTS_PER_CLUSTER: int   = 20

# Internal helpers

def _validate_inputs(
    target_object: np.ndarray,
    regions: dict[str, np.ndarray],
) -> None:
    if target_object.ndim != 2 or target_object.shape[1] != 4:
        raise ValueError(
            f"target_object must be (N, 4), got {target_object.shape}"
        )
    if not regions:
        raise ValueError("regions dict must not be empty")
    for name, pts in regions.items():
        if pts.ndim != 2 or pts.shape[1] != 4:
            raise ValueError(
                f"regions['{name}'] must be (K, 4), got {pts.shape}"
            )


def _sample_anchor_points(
    region_pts: np.ndarray,
    n_anchors: int,
    rng: np.random.Generator,
) -> np.ndarray:

    n = len(region_pts)
    if n == 0:
        return np.empty((0, 4), dtype=np.float32)
    replace = n < n_anchors
    idx     = rng.choice(n, size=n_anchors, replace=replace)
    return region_pts[idx].astype(np.float32)


def _generate_cluster(
    anchor: np.ndarray,
    cluster_size: int,
    std_m: float,
    rng: np.random.Generator,
) -> np.ndarray:

    xyz_offsets  = rng.normal(loc=0.0, scale=std_m, size=(cluster_size, 3))
    xyz          = anchor[:3] + xyz_offsets.astype(np.float32)

    # Intensity: anchor value ± 5 % jitter, clamped to [0, 1]
    intensity_jitter = rng.normal(loc=0.0, scale=0.05, size=(cluster_size, 1))
    intensity        = np.clip(anchor[3] + intensity_jitter, 0.0, 1.0).astype(np.float32)

    return np.concatenate([xyz, intensity], axis=1).astype(np.float32)


def _filter_by_surface_distance(
    candidates: np.ndarray,
    surface_pts: np.ndarray,
    max_dist_m: float,
) -> np.ndarray:

    if len(candidates) == 0 or len(surface_pts) == 0:
        return candidates

    cand_xyz  = candidates[:, :3].astype(np.float32)   # (C, 3)
    surf_xyz  = surface_pts[:, :3].astype(np.float32)  # (S, 3)

    # Compute pairwise squared distances in chunks to avoid huge memory spikes
    chunk      = 512
    min_dists  = np.full(len(cand_xyz), np.inf, dtype=np.float32)

    for start in range(0, len(surf_xyz), chunk):
        end   = min(start + chunk, len(surf_xyz))
        diff  = cand_xyz[:, None, :] - surf_xyz[None, start:end, :]   # (C, chunk, 3)
        sq    = (diff ** 2).sum(axis=2)                                 # (C, chunk)
        min_dists = np.minimum(min_dists, sq.min(axis=1))

    min_dists = np.sqrt(min_dists)
    return candidates[min_dists <= max_dist_m]



def adaptive_outlier_injection(
    target_object: np.ndarray,
    regions: dict[str, np.ndarray],
    n_fake_points: int = 200,
    injection_regions: Optional[list[str]] = None,
    max_dist_m: float = MAX_SURFACE_DIST_M,
    defense_aware: bool = False,
    n_clusters: Optional[int] = None,
    cluster_std_m: Optional[float] = None,
    pts_per_cluster: Optional[int] = None,
    collapse_centroid: Optional[np.ndarray] = None,
    collapse_pull: float = 0.5,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    
    _validate_inputs(target_object, regions)

    rng = np.random.default_rng(random_seed)

    # Resolve injection regions
    inj_regions = injection_regions if injection_regions is not None \
                  else DEFAULT_INJECTION_REGIONS
    # Keep only regions that actually exist in the dict
    inj_regions = [r for r in inj_regions if r in regions and len(regions[r]) > 0]

    if not inj_regions:
        if verbose:
            print("  [WARN] No valid injection regions found — returning original object.")
        return target_object.astype(np.float32)

    # Resolve cluster parameters based on defence mode
    _n_clusters      = n_clusters      if n_clusters      is not None \
                       else (DEFENSE_N_CLUSTERS      if defense_aware else DEFAULT_N_CLUSTERS)
    _std_m           = cluster_std_m   if cluster_std_m   is not None \
                       else (DEFENSE_CLUSTER_STD_M   if defense_aware else DEFAULT_CLUSTER_STD_M)
    _pts_per_cluster = pts_per_cluster if pts_per_cluster is not None \
                       else (DEFENSE_PTS_PER_CLUSTER if defense_aware else DEFAULT_PTS_PER_CLUSTER)

    # Distribute the fake-point budget evenly across injection regions
    pts_per_region = max(1, n_fake_points // len(inj_regions))
    remainder      = n_fake_points - pts_per_region * len(inj_regions)

    all_injected:   list[np.ndarray] = []
    region_reports: dict[str, dict]  = {}

    for i, region_name in enumerate(inj_regions):
        region_pts = regions[region_name].astype(np.float32)
        budget     = pts_per_region + (remainder if i == 0 else 0)

        # Compute how many "raw" candidates we need to generate to meet budget
        # after distance-filtering (generate 2× budget then filter, at minimum)
        raw_needed = max(budget * 2, _n_clusters * _pts_per_cluster)

        # Derive actual number of clusters for this region
        effective_clusters = max(1, raw_needed // max(_pts_per_cluster, 1))
        effective_clusters = max(effective_clusters, _n_clusters)

        # Sample anchor points from this region's surface
        anchors = _sample_anchor_points(region_pts, effective_clusters, rng)

        # If collapse_centroid is provided, pull anchors inward so
        # injected clusters fill the interior rather than the boundary
        if collapse_centroid is not None:
            cc = np.asarray(collapse_centroid, dtype=np.float32)[:3]
            for j in range(len(anchors)):
                anchor_xyz = anchors[j, :3]
                anchors[j, :3] = anchor_xyz + collapse_pull * (cc - anchor_xyz)

        # Generate clusters
        cluster_parts: list[np.ndarray] = []
        for anchor in anchors:
            cluster = _generate_cluster(anchor, _pts_per_cluster, _std_m, rng)
            cluster_parts.append(cluster)

        raw_candidates = np.concatenate(cluster_parts, axis=0).astype(np.float32)

        # Filter: keep only points within max_dist_m of ANY real surface point
        filtered = _filter_by_surface_distance(
            candidates  = raw_candidates,
            surface_pts = target_object,
            max_dist_m  = max_dist_m,
        )

        # Cap to per-region budget
        if len(filtered) > budget:
            idx      = rng.choice(len(filtered), size=budget, replace=False)
            filtered = filtered[idx]

        all_injected.append(filtered)
        region_reports[region_name] = {
            "budget":    budget,
            "generated": len(raw_candidates),
            "filtered":  len(filtered),
        }

    # Assemble final output
    injected_pts = (
        np.concatenate(all_injected, axis=0).astype(np.float32)
        if all_injected else np.empty((0, 4), dtype=np.float32)
    )

    attacked_object = np.concatenate(
        [target_object.astype(np.float32), injected_pts], axis=0
    ).astype(np.float32)

    if verbose:
        _print_injection_report(
            target_object    = target_object,
            attacked_object  = attacked_object,
            region_reports   = region_reports,
            inj_regions      = inj_regions,
            max_dist_m       = max_dist_m,
            defense_aware    = defense_aware,
            n_clusters       = _n_clusters,
            cluster_std_m    = _std_m,
            pts_per_cluster  = _pts_per_cluster,
        )

    return attacked_object



def _print_injection_report(
    target_object:   np.ndarray,
    attacked_object: np.ndarray,
    region_reports:  dict[str, dict],
    inj_regions:     list[str],
    max_dist_m:      float,
    defense_aware:   bool,
    n_clusters:      int,
    cluster_std_m:   float,
    pts_per_cluster: int,
) -> None:
    n_orig     = len(target_object)
    n_out      = len(attacked_object)
    n_injected = n_out - n_orig

    mode_str = "DEFENSE-AWARE (dense/tight)" if defense_aware else "NORMAL"
    print(f"\nAdaptive Outlier Injection")
    print(f"Mode: {mode_str}")
    #print(f"Clusters/region: {n_clusters}   σ = {cluster_std_m:.3f} m f"pts/cluster = {pts_per_cluster}")
    print(f"Max surface dist: {max_dist_m:.2f} m")
    print(f"  {'Region':<12} {'Budget':>8} {'Generated':>11} {'Injected':>10}")
    print(f"  {'─'*56}")

    for r in inj_regions:
        rr = region_reports.get(r, {})
        print(
            f"  {r:<12} {rr.get('budget', 0):>8,} "
            f"{rr.get('generated', 0):>11,} "
            f"{rr.get('filtered', 0):>10,}"
        )

    print(f"  {'─'*56}")
    print(f"  {'TOTAL':<12} {'':>8} {'':>11} {n_injected:>10,}")
    print(f"\n  Input shape : {target_object.shape}  →  "
          f"Output shape: {attacked_object.shape}")
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
        description="Submodule 4.1.3 — Adaptive Outlier Injection"
    )
    parser.add_argument(
        "--frame", type=int, required=True,
        help="0-based frame index to load from the dataset",
    )
    parser.add_argument(
        "--fake-points", type=int, default=200,
        help="Number of fake points to inject (default: 200)",
    )
    parser.add_argument(
        "--defense-aware", action="store_true", default=False,
        help="Enable defense-aware mode (denser, tighter clusters)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.1,
        help="Voxel size in metres for scene pre-processing (default: 0.1)",
    )
    args = parser.parse_args()

    print("\n\tSubmodule 4.1.3 — Adaptive Outlier Injection")
    print("-" * 60)

    # Load the target frame
    frames = get_all_frames(DATA_DIR)
    if not (0 <= args.frame < len(frames)):
        print(f"\n  [ERROR] Frame index {args.frame} out of range "
              f"(0–{len(frames)-1}).")
        sys.exit(1)

    frame_path = frames[args.frame]
    raw = load_bin_file(frame_path)

    print(f"\n Frame: {frame_path.name}")
    print(f"  Raw scene shape : {raw.shape}")

    # Extract only the largest vehicle object + its front/side/rear regions
    try:
        target_object, regions, _ = extract_target_and_regions(
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

    N = len(target_object)
    print(f"  Vehicle object  : {target_object.shape}  "
          f"(rear={len(regions['rear'])}, side={len(regions['side'])}, "
          f"front={len(regions['front'])})")

    # Run injection
    attacked = adaptive_outlier_injection(
        target_object = target_object,
        regions       = regions,
        n_fake_points = args.fake_points,
        defense_aware = args.defense_aware,
        verbose       = True,
        random_seed   = 42,
    )

    n_injected = len(attacked) - N

    print(f"\nOutput — Attacked object shape : {attacked.shape}")
    print(f"Injected points: {n_injected}")
    assert attacked.shape[1] == 4, "Output must have 4 columns"
    assert len(attacked) > N, "Output must have more points than input"

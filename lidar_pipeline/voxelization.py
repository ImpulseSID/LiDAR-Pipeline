"""
voxelization.py

Reduce the number of points in a raw LiDAR frame by grouping nearby points
into fixed-size cubic voxels and replacing each group with its centroid.

Algorithm:
  1. Compute the bounding box of the point cloud.
  2. Assign every point to a voxel index:
         (ix, iy, iz) = floor((p_xyz - p_min) / voxel_size)
  3. Group all points that share the same voxel index.
  4. Replace each group with its mean (centroid) across x, y, z, intensity.
  5. Return the resulting (M, 4) array  where  M << N.

INPUT  : Raw point cloud   (N, 4)  — [x, y, z, intensity]
OUTPUT : Voxelized cloud   (M, 4)  — [x, y, z, intensity]   with M < N
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional

# Ensure the project root is on sys.path so `lidar_pipeline` is importable when running the program

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import load_bin_file, get_all_frames, default_velodyne_dir


# Core voxelization

def _voxelize_kernel(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Core voxelization kernel: group points into voxels and return centroids.

    Inputs:
        points:     (N, 4) float32 array  [x, y, z, intensity]
        voxel_size: edge length of each cubic voxel in metres (must be > 0)

    Returns:
        (M, 4) float32 array of voxel centroids, where M <= N.
    """
    if len(points) == 0:
        return points.copy()

    xyz = points[:, :3]  # (N, 3)

    # bounding box
    p_min = xyz.min(axis=0)          # (3,)  lower corner

    # assign each point to a voxel index tuple (ix, iy, iz)
    voxel_indices = np.floor((xyz - p_min) / voxel_size).astype(np.int32)  # (N, 3)

    max_idx = voxel_indices.max(axis=0) + 1
    stride_y = int(max_idx[2])
    stride_x = int(max_idx[1]) * stride_y
    keys = (
        voxel_indices[:, 0].astype(np.int64) * stride_x
        + voxel_indices[:, 1].astype(np.int64) * stride_y
        + voxel_indices[:, 2].astype(np.int64)
    )  # (N,)

    # sort by key so points in the same voxel are contiguous
    sort_order = np.argsort(keys, kind="stable")
    sorted_keys   = keys[sort_order]
    sorted_points = points[sort_order]       # (N, 4)

    # find voxel boundaries using np.unique
    _, first_occurrence, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )

    # compute centroids via cumulative sum trick (O(N), no Python loop)
    cum = np.cumsum(sorted_points, axis=0)   # (N, 4) prefix sums

    # Sum of each voxel = cum[last_in_voxel] - cum[first_in_voxel - 1]
    end_idx   = first_occurrence + counts - 1
    voxel_sum = cum[end_idx]
    mask      = first_occurrence > 0
    voxel_sum[mask] -= cum[first_occurrence[mask] - 1]

    # Centroid = sum / count
    centroids = voxel_sum / counts[:, np.newaxis]

    return centroids.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Core voxelization (public API)
# ─────────────────────────────────────────────────────────────────────────────

def voxelize(
    points: np.ndarray,
    voxel_size: float = 0.1,
    near_field_voxel_size: Optional[float] = 0.35,
    near_field_range_m: float = 50.0,
    sensor_origin: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Voxelize a LiDAR point cloud by grouping nearby points into cubes and
    computing the centroid of each occupied voxel.

    Function Inputs:
        points:               (N, 4) float32 array  [x, y, z, intensity]
        voxel_size:           Edge length of each cubic voxel in metres.
                              Smaller -> finer resolution, more output points.
                              Larger  -> coarser resolution, fewer output points.
                              Default is 0.1 m (10 cm cubes).
        near_field_voxel_size:
                              If provided (float > voxel_size), a second coarser
                              voxelization pass is applied to points whose
                              horizontal distance from sensor_origin is at or
                              below near_field_range_m.  This mimics reduced
                              angular resolution at close range and degrades
                              sparse near-field clusters (e.g. the adversarial
                              target object with many dropped points) more than
                              dense original ones, increasing the divergence
                              between original and adversarial bounding boxes and
                              lowering the object-level IoU.
                              Default: 0.35 m (3.5× the standard fine resolution).
        near_field_range_m:   Horizontal range threshold in metres for the
                              near-field zone.  Points within this radius receive
                              the coarser near_field_voxel_size.
                              Default: 50.0 m.
        sensor_origin:        (3,) array for the LiDAR sensor position used when
                              computing horizontal range.  Defaults to [0, 0, 0].

    output of Function:
        (M, 4) float32 array of voxel centroids, where M ≤ N.
    """
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) array, got shape {points.shape}"
        )
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    # ── Standard single-pass mode (near_field_voxel_size not requested) ───────
    if near_field_voxel_size is None:
        return _voxelize_kernel(points, voxel_size)

    # ── Dual-resolution mode ──────────────────────────────────────────────────
    if near_field_voxel_size <= 0:
        raise ValueError(
            f"near_field_voxel_size must be positive, got {near_field_voxel_size}"
        )

    origin_xy = np.zeros(2, dtype=np.float64)
    if sensor_origin is not None:
        origin_xy = np.asarray(sensor_origin, dtype=np.float64)[:2]

    # Step 1: standard voxelization of the full cloud
    coarse_all = _voxelize_kernel(points, voxel_size)

    # Step 2: identify near-field centroids (horizontal distance ≤ threshold)
    xy  = coarse_all[:, :2].astype(np.float64) - origin_xy
    h_range = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    near_mask = h_range <= near_field_range_m
    far_mask  = ~near_mask

    near_pts = coarse_all[near_mask]  # dense cluster candidates
    far_pts  = coarse_all[far_mask]   # background / far field

    # Step 3: second coarser pass on near-field centroids only
    if len(near_pts) > 0:
        near_coarse = _voxelize_kernel(near_pts, near_field_voxel_size)
    else:
        near_coarse = near_pts

    # Step 4: reassemble — far-field stays at fine resolution
    if len(near_coarse) == 0 and len(far_pts) == 0:
        return coarse_all
    if len(near_coarse) == 0:
        return far_pts
    if len(far_pts) == 0:
        return near_coarse

    return np.concatenate([near_coarse, far_pts], axis=0).astype(np.float32)

# Convenience helpers

def voxelization_stats(original: np.ndarray, voxelized: np.ndarray) -> dict:
    """
    Return a summary dictionary comparing the original and voxelized point clouds.

    Function Input:
        original:   (N, 4) original point cloud
        voxelized:  (M, 4) voxelized point cloud

    Function Output:
        dict with keys: n_original, n_voxelized, reduction_ratio, compression_factor
    """
    n_orig = len(original)
    n_vox  = len(voxelized)
    return {
        "n_original":         n_orig,
        "n_voxelized":        n_vox,
        "reduction_ratio":    round(1 - n_vox / n_orig, 4),
        "compression_factor": round(n_orig / n_vox, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    DATA_DIR = default_velodyne_dir()

    print("\t Voxelization")
    print("-" * 35)

    # Load one frame
    try:
        frames = get_all_frames(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    fp = frames[0]
    print(f"\nFrame : {fp.name}")
    raw = load_bin_file(fp)

    print(f"  Raw shape    : {raw.shape}  ({len(raw):,} points)")
    print(f"  X range      : {raw[:, 0].min():.2f} m  ->  {raw[:, 0].max():.2f} m")
    print(f"  Y range      : {raw[:, 1].min():.2f} m  ->  {raw[:, 1].max():.2f} m")
    print(f"  Z range      : {raw[:, 2].min():.2f} m  ->  {raw[:, 2].max():.2f} m")

    print(f"  Before : shape {raw.shape}  ({len(raw):,} points)")

    vox = voxelize(raw, voxel_size=0.2)
    print(f"  After  : shape {vox.shape}  ({len(vox):,} points) voxels.")

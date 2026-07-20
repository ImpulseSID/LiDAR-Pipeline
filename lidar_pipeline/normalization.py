"""
normalization.py

Scale LiDAR point cloud coordinates into a fixed [0, 1] range so that
different scenes with large or shifted bounding boxes produce stable,
comparable inputs for neural-network training.
"""

import sys
import numpy as np
from pathlib import Path

# Ensure the project root is on sys.path so `lidar_pipeline` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import load_bin_file, get_all_frames, default_velodyne_dir
from lidar_pipeline.voxelization import voxelize


# Stateless helper

def normalize(points: np.ndarray) -> np.ndarray:

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) array, got shape {points.shape}"
        )

    out = points.copy().astype(np.float32)

    for axis in range(3):   # x=0, y=1, z=2  (skip intensity at index 3)
        col = out[:, axis]
        col_min = float(col.min())
        col_max = float(col.max())
        col_range = col_max - col_min

        if col_range == 0.0:
            # Degenerate: all points share the same coordinate on this axis
            out[:, axis] = 0.0
        else:
            out[:, axis] = (col - col_min) / col_range

    return out


# Stateful normalizer

class PointCloudNormalizer:

    def __init__(self) -> None:
        self.min_xyz:   np.ndarray | None = None
        self.max_xyz:   np.ndarray | None = None
        self.range_xyz: np.ndarray | None = None

    # fit

    def fit(self, points: np.ndarray) -> "PointCloudNormalizer":
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                f"Expected (N, 4) array, got shape {points.shape}"
            )

        xyz = points[:, :3]
        self.min_xyz   = xyz.min(axis=0).astype(np.float32)   # (3,)
        self.max_xyz   = xyz.max(axis=0).astype(np.float32)   # (3,)
        self.range_xyz = self.max_xyz - self.min_xyz            # (3,)

        # Prevent division-by-zero for degenerate axes
        self.range_xyz = np.where(
            self.range_xyz == 0.0,
            np.ones_like(self.range_xyz),   # safe sentinel -> normalized value = 0
            self.range_xyz,
        )

        return self

    # transform

    def transform(self, points: np.ndarray) -> np.ndarray:
        if self.min_xyz is None:
            raise RuntimeError(
                "PointCloudNormalizer has not been fitted. Call fit() first."
            )
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                f"Expected (N, 4) array, got shape {points.shape}"
            )

        out = points.copy().astype(np.float32)
        xyz_norm = (out[:, :3] - self.min_xyz) / self.range_xyz
        out[:, :3] = np.clip(xyz_norm, 0.0, 1.0)
        return out

    # fit_transform

    def fit_transform(self, points: np.ndarray) -> np.ndarray:
        return self.fit(points).transform(points)

    # repr

    def __repr__(self) -> str:
        if self.min_xyz is None:
            return "PointCloudNormalizer(unfitted)"
        return (
            f"PointCloudNormalizer("
            f"x=[{self.min_xyz[0]:.2f}, {self.max_xyz[0]:.2f}], "   # type: ignore[index]
            f"y=[{self.min_xyz[1]:.2f}, {self.max_xyz[1]:.2f}], "   # type: ignore[index]
            f"z=[{self.min_xyz[2]:.2f}, {self.max_xyz[2]:.2f}])"    # type: ignore[index]
        )


# CLI

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    DATA_DIR = default_velodyne_dir()

    print("\t Normalization")
    print("-" * 35)

    try:
        frames = get_all_frames(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    fp = frames[0]
    print(f"\nFrame : {fp.name}")
    raw = load_bin_file(fp)
    print(f"  Raw shape    : {raw.shape}  ({len(raw):,} points)")

    # Voxelization via import from voxelization.py
    vox = voxelize(raw, voxel_size=0.2)

    print(f"\n  After voxelization:")
    print(f"    Shape      : {vox.shape}  ({len(vox):,} voxel centroids)")
    print(f"    X range    : {vox[:, 0].min():.3f}  ->  {vox[:, 0].max():.3f}  m")
    print(f"    Y range    : {vox[:, 1].min():.3f}  ->  {vox[:, 1].max():.3f}  m")
    print(f"    Z range    : {vox[:, 2].min():.3f}  ->  {vox[:, 2].max():.3f}  m")
    print(f"    Intensity  : {vox[:, 3].min():.3f}  ->  {vox[:, 3].max():.3f}")

    # normalize
    norm = normalize(vox)

    print(f"\n  After normalization:")
    print(f"    Shape      : {norm.shape}")
    print(f"    X range    : {norm[:, 0].min():.6f}  ->  {norm[:, 0].max():.6f}")
    print(f"    Y range    : {norm[:, 1].min():.6f}  ->  {norm[:, 1].max():.6f}")
    print(f"    Z range    : {norm[:, 2].min():.6f}  ->  {norm[:, 2].max():.6f}")






"""
synchronization.py

Align sequential LiDAR frames for: temporal attack, tracking, trajectory analysis

Algorithm:
  Voxelize each raw frame to reduce point density.
  Compute the spatial centroid (mean x, y, z) of each voxelized frame.
  Compare every frame's centroid against frame_1's centroid to obtain a translation offset:  offset_i = centroid_1 - centroid_i
  Apply that offset to every point in frame_i, shifting it so all frames share the same reference origin (frame_1's centroid).
  Return the list of aligned (M_i, 4) arrays as the synchronized sequence.
"""

import sys
import numpy as np
from pathlib import Path

# Ensure the project root is on sys.path so `lidar_pipeline` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import load_bin_file, get_all_frames, default_velodyne_dir
from lidar_pipeline.voxelization import voxelize


# Core helpers

def compute_centroid(points: np.ndarray) -> np.ndarray:
    # Compute the spatial centroid (mean position) of a point cloud.
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) array, got shape {points.shape}")

    return points[:, :3].mean(axis=0).astype(np.float32)


def compute_offset(centroid_ref: np.ndarray, centroid_target: np.ndarray) -> np.ndarray:
    # Compare two centroids and return the translation needed to align the target frame onto the reference frame.

    return (centroid_ref - centroid_target).astype(np.float32)


def align_frame(points: np.ndarray, offset: np.ndarray) -> np.ndarray:
    # Translate a point cloud by a given (dx, dy, dz) offset. 
    aligned = points.copy().astype(np.float32)
    aligned[:, :3] += offset
    return aligned


# Core synchronization

def synchronize_frames(frames: list[np.ndarray], voxel_size: float = 0.2) -> list[np.ndarray]:
    # Align a sequence of raw LiDAR frames so they share a common spatial reference origin (the centroid of frame_1).

    if not frames:
        raise ValueError("frames list is empty — nothing to synchronize.")

    # Step 1 & 2: voxelize every frame and compute its centroid
    voxelized  = [voxelize(f, voxel_size=voxel_size) for f in frames]
    centroids  = [compute_centroid(v) for v in voxelized]

    # Reference is always frame_1 (index 0)
    centroid_ref = centroids[0]

    # Steps 3 & 4: compare each centroid to the reference and shift
    aligned = []
    for vox, centroid_i in zip(voxelized, centroids):
        offset   = compute_offset(centroid_ref, centroid_i)
        aligned.append(align_frame(vox, offset))

    return aligned


# Stateful synchronizer

class FrameSynchronizer:
    # Stateful wrapper around synchronize_frames().

    def __init__(self, voxel_size: float = 0.2) -> None:
        self.voxel_size    = voxel_size
        self.centroid_ref: np.ndarray | None = None

    def fit(self, reference_frame: np.ndarray) -> "FrameSynchronizer":
        """Set the reference centroid from a single reference frame."""
        vox = voxelize(reference_frame, voxel_size=self.voxel_size)
        self.centroid_ref = compute_centroid(vox)
        return self

    def transform(self, frame: np.ndarray) -> np.ndarray:
        """Align one frame to the stored reference centroid."""
        if self.centroid_ref is None:
            raise RuntimeError("FrameSynchronizer has not been fitted. Call fit() first.")

        vox    = voxelize(frame, voxel_size=self.voxel_size)
        offset = compute_offset(self.centroid_ref, compute_centroid(vox))
        return align_frame(vox, offset)

    def fit_transform(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Fit on the first frame and synchronize the full sequence."""
        self.fit(frames[0])
        return [self.transform(f) for f in frames]

    def __repr__(self) -> str:
        if self.centroid_ref is None:
            return "FrameSynchronizer(unfitted)"
        cx, cy, cz = self.centroid_ref
        return f"FrameSynchronizer(ref_centroid=[{cx:.2f}, {cy:.2f}, {cz:.2f}], voxel_size={self.voxel_size})"



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    DATA_DIR = default_velodyne_dir()

    print("\t Frame Synchronization")
    print("-" * 45)

    # Load three sequential frames
    try:
        all_frames = get_all_frames(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if len(all_frames) < 3:
        print(f"[ERROR] Need at least 3 frames, found {len(all_frames)}.")
        sys.exit(1)

    frame_paths = all_frames[:3]
    raw_frames  = [load_bin_file(fp) for fp in frame_paths]

    print(f"\nLoaded {len(raw_frames)} frames:")
    for i, (fp, f) in enumerate(zip(frame_paths, raw_frames), start=1):
        print(f"  frame_{i} : {fp.name}  ({len(f):,} points)")

    # Synchronize
    synced = synchronize_frames(raw_frames, voxel_size=0.2)

    # Report centroids before and after
    print("\nCentroid comparison (x, y, z)  [metres]:")
    print(f"  {'Frame':<10}  {'Before':>30}  {'After':>30}")
    print(f"  {'-'*10}  {'-'*30}  {'-'*30}")

    for i, (raw, aligned) in enumerate(zip(raw_frames, synced), start=1):
        c_before = voxelize(raw, voxel_size=0.2)[:, :3].mean(axis=0)
        c_after  = aligned[:, :3].mean(axis=0)
        print(
            f"  frame_{i:<5}  "
            f"[{c_before[0]:>7.2f}, {c_before[1]:>7.2f}, {c_before[2]:>7.2f}]  "
            f"[{c_after[0]:>7.2f}, {c_after[1]:>7.2f}, {c_after[2]:>7.2f}]"
        )

    print(f"\nOutput: {len(synced)} synchronized frames.")
    for i, s in enumerate(synced, start=1):
        print(f"  frame_{i} : shape {s.shape}  ({len(s):,} voxel centroids)")

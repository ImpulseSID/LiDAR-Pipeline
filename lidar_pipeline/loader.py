"""
loader.py

Converts KITTI .bin LiDAR frames into NumPy arrays.

KITTI LiDAR format:
  - Each point is stored as 4 consecutive float32 values: [x, y, z, intensity]
  - x: forward (m), y: left (m), z: up (m), intensity: [0.0, 1.0]
  - File: 000123.bin → numpy array of shape (N, 4)
"""

import os
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Dataset location (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
#
# The project migrated from a flat single-scene layout
#     data/velodyne/training/velodyne/*.bin
# to the KITTI *tracking* layout, where frames are grouped per sequence:
#     data/tracking_velodyne/training/training/velodyne/<seq>/NNNNNN.bin
#
# Paths are resolved relative to the project root (the parent of the
# ``lidar_pipeline`` package) so they work regardless of the current working
# directory.

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Root of the per-sequence tracking velodyne frames (note the doubled
# "training" produced when the dataset was unpacked in this project).
TRACKING_VELODYNE_ROOT: Path = (
    PROJECT_ROOT / "data" / "tracking_velodyne" / "training" / "training" / "velodyne"
)

# Sequence used by default when a script needs one concrete scene to operate on.
DEFAULT_SEQUENCE: str = "0000"


def default_velodyne_dir(seq: str = DEFAULT_SEQUENCE) -> Path:
    """Return the velodyne folder for one KITTI tracking sequence.

    The legacy scripts expect a *flat* directory of ``NNNNNN.bin`` frames; a
    single tracking sequence folder is exactly that, so pointing ``DATA_DIR`` at
    ``default_velodyne_dir(seq)`` keeps them working unchanged.
    """
    return TRACKING_VELODYNE_ROOT / seq


def list_tracking_sequences() -> list[str]:
    """Sorted list of available tracking sequence ids (e.g. ["0000", ...])."""
    if not TRACKING_VELODYNE_ROOT.exists():
        raise FileNotFoundError(
            f"Tracking velodyne root not found: {TRACKING_VELODYNE_ROOT}")
    return sorted(p.name for p in TRACKING_VELODYNE_ROOT.iterdir() if p.is_dir())


# Core loader

def load_bin_file(filepath: str | Path) -> np.ndarray:

    # Load a single KITTI LiDAR .bin file into a NumPy array.

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"LiDAR file not found: {filepath}")

    file_size = filepath.stat().st_size

    if file_size == 0:
        raise ValueError(f"Empty LiDAR file: {filepath}")

    # Each point = 4 float32 values = 4 × 4 bytes = 16 bytes
    if file_size % 16 != 0:
        raise ValueError(
            f"Corrupt LiDAR file: size {file_size} bytes is not a multiple of 16. "
            f"File: {filepath}"
        )

    # Read raw bytes and reshape into (N, 4)
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)

    return points


# Dataset helpers

def get_all_frames(data_dir: str | Path) -> list[Path]:

    # Scan a directory and return a sorted list of all .bin frame paths.

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    frames = sorted(data_dir.glob("*.bin"))

    if not frames:
        raise FileNotFoundError(f"No .bin files found in: {data_dir}")

    return frames


def load_frame_by_index(data_dir: str | Path, idx: int) -> np.ndarray:
    
    # Load a LiDAR frame by its index in the sorted file list.

    frames = get_all_frames(data_dir)

    if not (0 <= idx < len(frames)):
        raise IndexError(
            f"Frame index {idx} out of range. Dataset has {len(frames)} frames (0 to {len(frames) - 1})."
        )

    return load_bin_file(frames[idx])


def get_frame_stats(points: np.ndarray) -> dict:

    # Compute basic statistics for a loaded point cloud.

    return {
        "num_points":      len(points),
        "x_range":         (float(points[:, 0].min()), float(points[:, 0].max())),
        "y_range":         (float(points[:, 1].min()), float(points[:, 1].max())),
        "z_range":         (float(points[:, 2].min()), float(points[:, 2].max())),
        "intensity_range": (float(points[:, 3].min()), float(points[:, 3].max())),
    }


if __name__ == "__main__":
    import sys

    DATA_DIR = default_velodyne_dir()

    print("Read Binary LiDAR File")
    print("_" * 40)

    # List available frames
    try:
        frames = get_all_frames(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"Found {len(frames)} LiDAR frames in: {DATA_DIR}")
    print(f"First frame : {frames[0].name}")
    print(f"Last frame  : {frames[-1].name}")

    # Load the first frame and display stats
    print("\nStats of First Frame/File:")
    print("-" * 40)
    points = load_bin_file(frames[0])

    print(f"Output shape : {points.shape}")
    print(f"Output dtype : {points.dtype}")

    stats = get_frame_stats(points)
    print(f"\nPoint cloud statistics:")
    print(f"  Total points : {stats['num_points']:,}")
    print(f"  X range (fwd): {stats['x_range'][0]:.2f} m  ->  {stats['x_range'][1]:.2f} m")
    print(f"  Y range (lat): {stats['y_range'][0]:.2f} m  ->  {stats['y_range'][1]:.2f} m")
    print(f"  Z range (up) : {stats['z_range'][0]:.2f} m  ->  {stats['z_range'][1]:.2f} m")
    print(f"  Intensity    : {stats['intensity_range'][0]:.3f}  ->  {stats['intensity_range'][1]:.3f}")

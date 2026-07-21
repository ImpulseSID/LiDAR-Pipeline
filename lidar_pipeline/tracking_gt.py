"""
tracking_gt.py

Task 3 — Ground-truth tracks for the tracking-impact evaluation.

Task 3's metrics compare the tracker output against ground truth:
  * Velocity Error   needs the object's true velocity.
  * Trajectory Error needs the object's true (x, y) path.
  * ID Switches / Fragmentation are measured by matching tracker hypotheses to
    ground-truth objects frame by frame.

This module parses the KITTI *tracking* ground truth (``label_02/<seq>.txt``)
together with the calibration (``calib/<seq>.txt``), converts every object
centre from the camera/rect frame (where KITTI stores labels) into the
**velodyne / LiDAR frame** (where PointPillars produces its detections), and
derives per-track ground-truth velocity, speed and trajectory.

KITTI tracking label columns (space separated, one row per object per frame):
     0  frame        integer frame index
     1  track_id     unique object id (-1 for DontCare)
     2  type         Car / Van / Pedestrian / Cyclist / DontCare / ...
     3  truncated
     4  occluded
     5  alpha
     6-9  bbox 2D    left top right bottom (image pixels)
    10  h            3D box height   (camera frame, metres)
    11  w            3D box width
    12  l            3D box length
    13  cx           location X      (camera/rect frame)
    14  cy           location Y
    15  cz           location Z
    16  rotation_y   yaw about the camera Y axis

Calibration (velodyne -> rectified camera):
    x_cam = R_rect @ Tr_velo_cam @ [x_velo, y_velo, z_velo, 1]^T
We invert this to place GT centres in the velodyne frame.

Output: a tidy ``pandas.DataFrame``, one row per (frame, track_id):
    frame, track_id, cls, x, y, z, h, w, l, ry, vx, vy, speed
with (x, y, z) in the velodyne frame and (vx, vy, speed) in m/s (KITTI is
10 Hz, so dt = 0.1 s).
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

# Ensure the project root is on sys.path so `lidar_pipeline` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import PROJECT_ROOT, DEFAULT_SEQUENCE


# ─────────────────────────────────────────────────────────────────────────────
# Dataset locations (KITTI tracking layout used in this project)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_ROOT: Path = (
    PROJECT_ROOT / "data" / "data_tracking_label_2" / "training" / "label_02"
)
CALIB_ROOT: Path = (
    PROJECT_ROOT / "data" / "data_tracking_calib" / "training" / "calib"
)

# KITTI captures at 10 Hz.
FRAME_RATE_HZ: float = 10.0
DT: float = 1.0 / FRAME_RATE_HZ

# Object types kept as real tracked objects (DontCare and anything else dropped).
VALID_TYPES: tuple[str, ...] = ("Car", "Van", "Truck", "Pedestrian", "Cyclist")

# Column layout of the raw KITTI tracking label file.
_LABEL_COLUMNS = [
    "frame", "track_id", "type", "truncated", "occluded", "alpha",
    "bbox_l", "bbox_t", "bbox_r", "bbox_b",
    "h", "w", "l", "cx", "cy", "cz", "ry",
]

# Final tidy schema.
GT_COLUMNS = [
    "frame", "track_id", "cls", "x", "y", "z",
    "h", "w", "l", "ry", "vx", "vy", "speed",
]


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_calib(
    seq: str = DEFAULT_SEQUENCE,
    calib_root: Union[str, Path] = CALIB_ROOT,
) -> dict:
    """Parse a KITTI tracking calib file into numpy matrices.

    Returns ``P2`` (3x4), ``R_rect`` (3x3), ``Tr_velo_cam`` (3x4) plus the
    convenience 4x4 homogeneous transforms ``velo_to_cam`` and ``cam_to_velo``.
    """
    calib_path = Path(calib_root) / f"{seq}.txt"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calib file not found: {calib_path}")

    raw: dict[str, np.ndarray] = {}
    for line in calib_path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, rest = line.partition(":")
        if rest:
            values = rest.split()
        else:
            # tracking calib keys like "R_rect" may have no colon
            parts = line.split()
            key, values = parts[0], parts[1:]
        try:
            raw[key.strip()] = np.array([float(v) for v in values], dtype=np.float64)
        except ValueError:
            continue

    def _reshape(key: str, rows: int, cols: int) -> np.ndarray:
        if key not in raw:
            raise KeyError(f"Calib key '{key}' missing in {calib_path}")
        return raw[key].reshape(rows, cols)

    P2 = _reshape("P2", 3, 4)
    R_rect = _reshape("R_rect", 3, 3)
    Tr_velo_cam = _reshape("Tr_velo_cam", 3, 4)

    # 4x4 velodyne -> rectified-camera:  x_cam = R_rect @ Tr_velo_cam @ x_velo_h
    R_rect_h = np.eye(4, dtype=np.float64)
    R_rect_h[:3, :3] = R_rect
    Tr_h = np.eye(4, dtype=np.float64)
    Tr_h[:3, :4] = Tr_velo_cam
    velo_to_cam = R_rect_h @ Tr_h
    cam_to_velo = np.linalg.inv(velo_to_cam)

    return {
        "P2": P2,
        "R_rect": R_rect,
        "Tr_velo_cam": Tr_velo_cam,
        "velo_to_cam": velo_to_cam,
        "cam_to_velo": cam_to_velo,
    }


def camera_to_velo(points_cam: np.ndarray, calib: dict) -> np.ndarray:
    """Transform (N,3) points from the rectified-camera frame to velodyne frame."""
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    return (calib["cam_to_velo"] @ hom.T).T[:, :3]


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth track parsing
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_tracks(
    seq: str = DEFAULT_SEQUENCE,
    label_root: Union[str, Path] = LABEL_ROOT,
    calib_root: Union[str, Path] = CALIB_ROOT,
    max_frame: int | None = None,
    valid_types: tuple[str, ...] = VALID_TYPES,
) -> pd.DataFrame:
    """Load ground-truth tracks for one sequence, in the velodyne frame.

    Parameters
    ----------
    seq        : sequence id, e.g. "0000".
    max_frame  : if given, keep only frames with index <= max_frame (e.g. clip
                 to the adversarial window).
    valid_types: object types to keep (DontCare and others dropped).

    Returns a DataFrame with columns ``GT_COLUMNS``; (x, y, z) in the velodyne
    frame and (vx, vy, speed) finite-difference velocities in the ground plane.
    """
    label_path = Path(label_root) / f"{seq}.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    calib = load_calib(seq, calib_root)

    df = pd.read_csv(
        label_path, sep=r"\s+", header=None, names=_LABEL_COLUMNS,
        engine="python",
    )

    # Drop DontCare / unwanted classes and invalid track ids.
    df = df[df["type"].isin(valid_types)].copy()
    df = df[df["track_id"] >= 0].copy()
    if max_frame is not None:
        df = df[df["frame"] <= max_frame].copy()

    if df.empty:
        return pd.DataFrame(columns=GT_COLUMNS)

    # Convert camera-frame centres to velodyne frame.
    velo = camera_to_velo(df[["cx", "cy", "cz"]].to_numpy(dtype=np.float64), calib)
    df["x"], df["y"], df["z"] = velo[:, 0], velo[:, 1], velo[:, 2]
    df["cls"] = df["type"]

    out = df[["frame", "track_id", "cls", "x", "y", "z", "h", "w", "l", "ry"]].copy()
    out = out.sort_values(["track_id", "frame"]).reset_index(drop=True)

    # Finite-difference velocity per track (ground plane). First frame -> 0.
    out["vx"] = 0.0
    out["vy"] = 0.0
    for _, grp in out.groupby("track_id"):
        idx = grp.index
        out.loc[idx, "vx"] = (grp["x"].diff() / DT).fillna(0.0).to_numpy()
        out.loc[idx, "vy"] = (grp["y"].diff() / DT).fillna(0.0).to_numpy()
    out["speed"] = np.sqrt(out["vx"] ** 2 + out["vy"] ** 2)

    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)


def gt_trajectory(gt: pd.DataFrame, track_id: int) -> pd.DataFrame:
    """Return the (frame, x, y) trajectory for one ground-truth track."""
    traj = gt[gt["track_id"] == track_id][["frame", "x", "y"]]
    return traj.sort_values("frame").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 3 — load KITTI GT tracks")
    parser.add_argument("--seq", default=DEFAULT_SEQUENCE, help="sequence id (default 0000)")
    parser.add_argument("--max-frame", type=int, default=None,
                        help="clip to frames <= this index")
    args = parser.parse_args()

    gt = load_gt_tracks(seq=args.seq, max_frame=args.max_frame)
    print(f"Loaded {len(gt)} GT rows | "
          f"{gt['frame'].nunique()} frames | "
          f"{gt['track_id'].nunique()} tracks | "
          f"classes: {sorted(gt['cls'].unique())}")
    print()
    print(gt.head(12).to_string(index=False))
    print("\nPer-track summary:")
    for tid, g in gt.groupby("track_id"):
        print(f"  track {tid:>2} ({g['cls'].iloc[0]:<10}): "
              f"{len(g):>3} frames, frames {g['frame'].min()}–{g['frame'].max()}, "
              f"mean speed {g['speed'].mean():.2f} m/s")

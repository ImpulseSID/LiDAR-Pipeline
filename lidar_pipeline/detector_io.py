"""
detector_io.py

Task 3 — I/O contract between the (Colab, GPU) PointPillars detector and the
(local / CPU) tracking + metrics pipeline.

The detector runs OpenPCDet PointPillars over each LiDAR frame and writes ONE
CSV per (sequence, condition). Keeping detector output as a small plain CSV is
deliberate: the heavy GPU step emits a few hundred rows that the rest of the
pipeline consumes anywhere, with no torch / pcdet dependency.

Detection CSV schema (one row per detected box) — matches the PDF's required
"Frame ID, Bounding Box, Confidence, Center Coordinates":
    frame  : int    0-based frame index (matches the source .bin number)
    x, y, z: float  box CENTRE in the VELODYNE / LiDAR frame (metres)
    dx,dy,dz: float box size (length, width, height) in the LiDAR frame
    yaw    : float  heading about the vertical axis (radians)
    score  : float  detection confidence in [0, 1]
    cls    : str    class name (Car / Pedestrian / Cyclist / Van / ...)

Object/Track ID is intentionally NOT here — identities are assigned by the
trackers, so they live in the tracking-result CSV, not the detection CSV.

This module never imports torch or pcdet, so it is safe to use locally.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DETECTION_COLUMNS = ["frame", "x", "y", "z", "dx", "dy", "dz", "yaw", "score", "cls"]

# Box-only columns (the geometric state a tracker consumes): [x,y,z,dx,dy,dz,yaw].
BOX_COLUMNS = ["x", "y", "z", "dx", "dy", "dz", "yaw"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-(sequence, condition) file naming
# ─────────────────────────────────────────────────────────────────────────────

def detections_path(out_dir: Union[str, Path], seq: str, condition: str) -> Path:
    """Standard path for a detection CSV, e.g. clean_detections_0000.csv.

    condition is "clean" or "adv"; seq is the sequence id ("0000", ...).
    """
    if condition not in ("clean", "adv"):
        raise ValueError(f"condition must be 'clean' or 'adv', got {condition!r}")
    return Path(out_dir) / f"{condition}_detections_{seq}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Load / save / validate
# ─────────────────────────────────────────────────────────────────────────────

def empty_detections() -> pd.DataFrame:
    """An empty detections DataFrame with the correct columns."""
    df = pd.DataFrame({c: [] for c in DETECTION_COLUMNS})
    return df.astype({c: "float64" for c in DETECTION_COLUMNS if c != "cls"}
                     | {"frame": "int64", "cls": "object"})


def save_detections(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write a detections DataFrame to CSV, validating the schema first."""
    missing = [c for c in DETECTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Detections missing columns: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[DETECTION_COLUMNS].to_csv(path, index=False)
    return path


def load_detections(path: Union[str, Path]) -> pd.DataFrame:
    """Load a detections CSV written by the Colab detector."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Detections CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in DETECTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Detections CSV {path} missing columns: {missing}")
    df["frame"] = df["frame"].astype(int)
    df["cls"] = df["cls"].astype(str)
    return (df.sort_values(["frame", "score"], ascending=[True, False])
              .reset_index(drop=True))


def filter_detections(
    df: pd.DataFrame,
    score_thresh: float = 0.0,
    classes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Filter detections by minimum score and/or an allowed class set."""
    out = df[df["score"] >= score_thresh]
    if classes is not None:
        out = out[out["cls"].isin(classes)]
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Frame access helpers
# ─────────────────────────────────────────────────────────────────────────────

def frames_in(df: pd.DataFrame) -> list[int]:
    """Sorted unique frame indices present in the detections."""
    return sorted(int(f) for f in df["frame"].unique())


def detections_for_frame(df: pd.DataFrame, frame: int) -> pd.DataFrame:
    """All detections for one frame, highest score first."""
    return (df[df["frame"] == frame]
            .sort_values("score", ascending=False)
            .reset_index(drop=True))


def to_frame_list(df: pd.DataFrame, frame_range: range | None = None) -> list[dict]:
    """Group detections into a per-frame list for streaming into a tracker.

    Returns a list of dicts (ordered by frame):
        {"frame": int,
         "boxes":   np.ndarray (M,7)  [x,y,z,dx,dy,dz,yaw],
         "scores":  np.ndarray (M,),
         "classes": list[str]}

    When ``frame_range`` is given, frames with NO detections are still included
    (empty arrays), so trackers see the true temporal gaps — essential for
    correct track-death and fragmentation behaviour.
    """
    frames = list(frame_range) if frame_range is not None else frames_in(df)
    out: list[dict] = []
    for f in frames:
        sub = detections_for_frame(df, f)
        out.append({
            "frame": int(f),
            "boxes": sub[BOX_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 7),
            "scores": sub["score"].to_numpy(dtype=np.float64),
            "classes": sub["cls"].tolist(),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic detections (for local pipeline testing without a GPU/detector)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_detections(
    n_frames: int = 20,
    seed: int = 0,
    drop_frames: tuple[int, ...] = (),
    noise: float = 0.0,
) -> pd.DataFrame:
    """Deterministic fake detections for testing the tracking pipeline.

    Two objects move in straight lines. ``drop_frames`` omits all detections on
    those frames (to exercise fragmentation); ``noise`` adds gaussian jitter to
    the centres (to simulate an attack degrading localisation).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        if f in drop_frames:
            continue
        ax, ay = 10.0 + 1.5 * f, 2.0          # a car driving forward
        bx, by = 8.0, -5.0 + 0.4 * f          # a pedestrian crossing
        for (cx, cy, dx, dy, dz, cls) in [
            (ax, ay, 4.2, 1.8, 1.5, "Car"),
            (bx, by, 0.8, 0.7, 1.7, "Pedestrian"),
        ]:
            jit = rng.normal(0.0, noise, size=2) if noise > 0 else np.zeros(2)
            rows.append({
                "frame": f,
                "x": cx + jit[0], "y": cy + jit[1], "z": -1.6,
                "dx": dx, "dy": dy, "dz": dz, "yaw": 0.0,
                "score": 0.9, "cls": cls,
            })
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


if __name__ == "__main__":
    # Self-test: round-trip + per-frame grouping with a fragmentation gap.
    import tempfile

    df = make_synthetic_detections(n_frames=10, drop_frames=(4, 5), noise=0.05)
    print("Synthetic detections:")
    print(df.to_string(index=False))

    tmp = Path(tempfile.gettempdir()) / "detio_selftest.csv"
    save_detections(df, tmp)
    reloaded = load_detections(tmp)
    assert list(reloaded.columns) == DETECTION_COLUMNS, "schema mismatch on reload"
    assert len(reloaded) == len(df), "row count changed on round-trip"
    tmp.unlink(missing_ok=True)

    print("\nframes present:", frames_in(df))
    fl = to_frame_list(df, frame_range=range(10))
    print("per-frame box counts (range 0-9, gaps preserved):",
          [len(x["boxes"]) for x in fl])
    print("path helper:", detections_path("outputs", "0000", "clean").name)
    print("\nround-trip + grouping OK")

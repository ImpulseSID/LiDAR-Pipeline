"""
visualize_attack.py

Module 4 — Visualise the POPA attack with Open3D.

Shows ONLY the attacked objects (the vehicles), not the rest of the scene. For a
chosen frame it detects the vehicles in the original frame, crops both the
original and the adversarial frame to those vehicle bounding boxes, and renders
them so you can see exactly which parts of each vehicle the attack removed.

Two modes:
  compare  (default) — original vehicles (green) beside the adversarial
                       vehicles (red), offset so you can compare shapes.
  overlay            — in place: points the attack REMOVED are red, points that
                       SURVIVED are green (the clearest view of the attack).

Usage:
    python -m lidar_pipeline.visualize_attack --frame 0
    python -m lidar_pipeline.visualize_attack --frame 6 --mode overlay
    python -m lidar_pipeline.visualize_attack --frame 8 --adv-dir data/adversarial
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import load_bin_file, default_velodyne_dir, DEFAULT_SEQUENCE
from lidar_pipeline.attack import extract_all_vehicles, BBOX_PAD_M


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ORIG_DIR = default_velodyne_dir(DEFAULT_SEQUENCE)
DEFAULT_ADV_DIR = Path("data/adversarial")

COLOR_ORIG = [0.15, 0.80, 0.25]   # green  — original / surviving points
COLOR_ADV = [0.90, 0.15, 0.15]    # red    — adversarial / removed points


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_to_vehicles(
    frame: np.ndarray,
    vehicles: list[dict],
    pad_m: float = BBOX_PAD_M,
) -> np.ndarray:
    """Keep only frame points inside any vehicle's (padded) bounding box."""
    xyz = frame[:, :3]
    keep = np.zeros(len(frame), dtype=bool)
    for v in vehicles:
        lo = v["bbox_min"] - pad_m
        hi = v["bbox_max"] + pad_m
        keep |= np.all((xyz >= lo) & (xyz <= hi), axis=1)
    return frame[keep]


def split_removed_kept(
    orig_obj: np.ndarray,
    adv_obj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split original object points into (removed, kept).

    The adversarial frame is a strict subset of the original, so a point was
    kept iff it still appears in the adversarial cloud (exact float match).
    """
    if len(orig_obj) == 0:
        return orig_obj, orig_obj
    adv_set = set(map(tuple, adv_obj.tolist()))
    kept_mask = np.array([tuple(r) in adv_set for r in orig_obj.tolist()], dtype=bool)
    return orig_obj[~kept_mask], orig_obj[kept_mask]


def _make_pcd(xyz: np.ndarray, color: list[float]):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3].astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def _make_boxes(vehicles: list[dict], color: list[float], offset: np.ndarray):
    import open3d as o3d
    boxes = []
    for v in vehicles:
        lo = v["bbox_min"].astype(np.float64) + offset
        hi = v["bbox_max"].astype(np.float64) + offset
        box = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        box.color = color
        boxes.append(box)
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize(
    orig_frame: np.ndarray,
    adv_frame: np.ndarray,
    vehicles: list[dict],
    mode: str = "compare",
    pad_m: float = BBOX_PAD_M,
) -> None:
    import open3d as o3d

    orig_obj = crop_to_vehicles(orig_frame, vehicles, pad_m)
    adv_obj = crop_to_vehicles(adv_frame, vehicles, pad_m)

    removed, kept = split_removed_kept(orig_obj, adv_obj)
    print(f"  Vehicles shown     : {len(vehicles)}")
    print(f"  Original obj points: {len(orig_obj):,}")
    print(f"  Surviving points   : {len(kept):,}")
    print(f"  Removed points     : {len(removed):,}  "
          f"({100.0 * len(removed) / max(len(orig_obj), 1):.1f}%)")

    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)]
    zero = np.zeros(3)

    if mode == "overlay":
        # In place: removed = red, surviving = green.
        if len(kept):
            geoms.append(_make_pcd(kept, COLOR_ORIG))
        if len(removed):
            geoms.append(_make_pcd(removed, COLOR_ADV))
        geoms += _make_boxes(vehicles, [0.4, 0.4, 0.4], zero)
        title = "POPA overlay — red = removed, green = surviving"
    else:
        # Side by side: original (green) | adversarial survivors (red), offset in Y.
        if len(orig_obj):
            span_y = float(orig_obj[:, 1].max() - orig_obj[:, 1].min())
        else:
            span_y = 20.0
        offset = np.array([0.0, span_y + 5.0, 0.0])

        if len(orig_obj):
            geoms.append(_make_pcd(orig_obj, COLOR_ORIG))
            geoms += _make_boxes(vehicles, COLOR_ORIG, zero)
        if len(adv_obj):
            shifted = adv_obj.copy()
            shifted[:, 1] += offset[1]
            geoms.append(_make_pcd(shifted, COLOR_ADV))
            geoms += _make_boxes(vehicles, COLOR_ADV, offset)
        title = "POPA compare — left/green = original, right/red = adversarial"

    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1400,
        height=900,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Module 4 — Visualise POPA attacked objects (Open3D)"
    )
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame number to view (matches NNNNNN.bin / adv_NNNN.bin)")
    parser.add_argument("--orig-dir", type=str, default=str(DEFAULT_ORIG_DIR),
                        help="Directory of original NNNNNN.bin frames")
    parser.add_argument("--adv-dir", type=str, default=str(DEFAULT_ADV_DIR),
                        help="Directory of adversarial adv_NNNN.bin frames")
    parser.add_argument("--mode", choices=["compare", "overlay"], default="compare",
                        help="compare = side by side; overlay = removed vs kept in place")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="Voxel size for vehicle detection (default: 0.1)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    try:
        import open3d  # noqa: F401
    except ImportError:
        print("  [ERROR] Open3D is not installed. Install it with:\n"
              "      pip install open3d")
        return 1

    orig_path = Path(args.orig_dir) / f"{args.frame:06d}.bin"
    adv_path = Path(args.adv_dir) / f"adv_{args.frame:04d}.bin"

    print("\n\tModule 4 — Attack Visualisation")
    print("-" * 60)

    if not orig_path.exists():
        print(f"  [ERROR] Original frame not found: {orig_path}")
        return 1
    if not adv_path.exists():
        print(f"  [ERROR] Adversarial frame not found: {adv_path}")
        return 1

    print(f"  Original    : {orig_path.name}")
    print(f"  Adversarial : {adv_path.name}")
    print(f"  Mode        : {args.mode}")

    orig_frame = load_bin_file(orig_path)
    adv_frame = load_bin_file(adv_path)

    vehicles = extract_all_vehicles(orig_frame, voxel_size=args.voxel_size)
    if not vehicles:
        print("  [ERROR] No vehicles detected in the original frame — nothing to show.")
        return 1

    visualize(orig_frame, adv_frame, vehicles, mode=args.mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

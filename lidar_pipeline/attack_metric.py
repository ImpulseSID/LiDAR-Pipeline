"""
attack_metric.py

Module 4 — Attack Success Metrics (Original vs Adversarial)

For every (original, adversarial) frame pair this script detects the target
vehicle in both frames and reports two metrics:

  * IoU            — 3D axis-aligned bounding-box IoU of the target vehicle,
                     original box vs adversarial box. Lower = the detector's
                     box moved / shrank more = stronger attack.
  * Confidence Drop — fall in the target's detection confidence:
                          conf = surviving points / original points
                          drop = conf_original - conf_adversarial
                     Confidence is normalised by the target object's OWN full
                     point count (not a fixed cap), so the drop reflects how
                     much of that specific object the attack destroyed. Higher
                     = the detector has far less evidence = stronger attack.

The detector is the same simulated pipeline used elsewhere in the project
(voxelize → semantic segmentation → DBSCAN instance segmentation → vehicle
clusters).

An attack on a frame is a SUCCESS only if BOTH hold:
    IoU            < 0.55
    Confidence Drop > 0.75

Usage:
    python -m lidar_pipeline.attack_metric
    python -m lidar_pipeline.attack_metric --seq 0000 --adv-dir data/adversarial
"""

from __future__ import annotations

import re
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import load_bin_file, default_velodyne_dir, DEFAULT_SEQUENCE
from lidar_pipeline.voxelization import voxelize
from lidar_pipeline.semantic_segmentation import segment_point_cloud, CLASS_VEHICLE
from lidar_pipeline.instance_segmentation import instance_segment, cluster_summary


# ─────────────────────────────────────────────────────────────────────────────
# Success thresholds  (both must be satisfied)
# ─────────────────────────────────────────────────────────────────────────────

IOU_SUCCESS_MAX: float = 0.55        # IoU must be strictly below this
CONF_DROP_SUCCESS_MIN: float = 0.75  # confidence drop must be strictly above this

# Detector settings
VOXEL_SIZE: float = 0.1
CONF_NORM: float = 200.0             # confidence = min(1, n_points / CONF_NORM)
MATCH_GATE_M: float = 3.0            # max centre distance to match orig↔adv object

# Default directories
DEFAULT_ORIG_DIR = default_velodyne_dir(DEFAULT_SEQUENCE)
DEFAULT_ADV_DIR = Path("data/adversarial")


# ─────────────────────────────────────────────────────────────────────────────
# Detection (simulated detector, consistent with tracking_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def detect_vehicles(frame: np.ndarray, voxel_size: float = VOXEL_SIZE) -> list[dict]:
    """
    Detect vehicle instances in a raw LiDAR frame.

    Returns a list of detection dicts with keys:
        center, dims, bbox_min, bbox_max, n_points, confidence
    """
    vox = voxelize(frame, voxel_size=voxel_size)
    sem_labels = segment_point_cloud(vox)
    inst_ids = instance_segment(vox, sem_labels)

    rows = cluster_summary(sem_labels, inst_ids)
    detections: list[dict] = []

    for row in rows:
        if row["semantic_class"] != CLASS_VEHICLE:
            continue
        mask = inst_ids == row["instance_id"]
        pts = vox[mask]
        if len(pts) < 3:
            continue

        xyz = pts[:, :3].astype(np.float64)
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)

        detections.append({
            "center": (lo + hi) / 2.0,
            "dims": hi - lo,
            "bbox_min": lo,
            "bbox_max": hi,
            "n_points": int(row["n_points"]),
            "confidence": min(1.0, row["n_points"] / CONF_NORM),
        })

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_3d_iou(
    min_a: np.ndarray, max_a: np.ndarray,
    min_b: np.ndarray, max_b: np.ndarray,
) -> float:
    """Axis-aligned 3D bounding-box IoU."""
    lo = np.maximum(min_a, min_b)
    hi = np.minimum(max_a, max_b)
    overlap = np.maximum(hi - lo, 0.0)
    inter = float(np.prod(overlap))

    vol_a = float(np.prod(np.maximum(max_a - min_a, 1e-9)))
    vol_b = float(np.prod(np.maximum(max_b - min_b, 1e-9)))
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def _match_target(target: dict, candidates: list[dict]) -> Optional[dict]:
    """
    Find the adversarial detection corresponding to the original target.

    Prefer the highest-IoU candidate; if none overlaps, fall back to the nearest
    centre within MATCH_GATE_M. Returns None if the object is undetectable in the
    adversarial frame (a maximal attack — full confidence loss).
    """
    if not candidates:
        return None

    best_iou, best_by_iou = 0.0, None
    for c in candidates:
        iou = compute_3d_iou(
            target["bbox_min"], target["bbox_max"],
            c["bbox_min"], c["bbox_max"],
        )
        if iou > best_iou:
            best_iou, best_by_iou = iou, c

    if best_by_iou is not None:
        return best_by_iou

    # No overlap: nearest centre within gate
    dists = [float(np.linalg.norm(target["center"] - c["center"])) for c in candidates]
    j = int(np.argmin(dists))
    return candidates[j] if dists[j] <= MATCH_GATE_M else None


def evaluate_pair(
    orig_path: Path,
    adv_path: Path,
    voxel_size: float = VOXEL_SIZE,
) -> Optional[dict]:
    """
    Compute IoU and confidence drop for one (original, adversarial) frame pair.

    Returns a metrics dict, or None if no target vehicle exists in the original
    frame (nothing to attack / measure). Cooldown frames (adversarial file is
    byte-identical to the original) are flagged with "clean": True and returned
    without running the expensive detector, since there is no attack to measure.
    """
    orig_arr = load_bin_file(orig_path)
    adv_arr = load_bin_file(adv_path)

    # Cheap short-circuit: an untouched (cooldown) frame is a copy of the
    # original, so IoU=1 and confidence drop=0 by definition — skip detection.
    if orig_arr.shape == adv_arr.shape and np.array_equal(orig_arr, adv_arr):
        return {
            "orig": orig_path.name, "adv": adv_path.name,
            "iou": 1.0, "conf_orig": None, "conf_adv": None, "conf_drop": 0.0,
            "n_orig": 0, "n_adv": 0,
            "iou_ok": False, "drop_ok": False, "success": False, "clean": True,
        }

    orig_dets = detect_vehicles(orig_arr, voxel_size)
    adv_dets = detect_vehicles(adv_arr, voxel_size)

    if not orig_dets:
        return None

    # Target = most confident (largest) vehicle in the original frame.
    target = max(orig_dets, key=lambda d: d["n_points"])
    match = _match_target(target, adv_dets)

    # Confidence is measured relative to the target's OWN full point count, not
    # a fixed cap — otherwise any object with >200 points saturates at 1.0 and
    # the drop is masked. conf = surviving points / original points, so the
    # drop directly reflects how much of THIS object the attack destroyed.
    n_orig = target["n_points"]
    n_adv = 0 if match is None else match["n_points"]

    conf_orig = 1.0
    conf_adv = float(np.clip(n_adv / max(n_orig, 1), 0.0, 1.0))

    if match is None:
        iou = 0.0
    else:
        iou = compute_3d_iou(
            target["bbox_min"], target["bbox_max"],
            match["bbox_min"], match["bbox_max"],
        )

    conf_drop = float(np.clip(conf_orig - conf_adv, 0.0, 1.0))

    success = (iou < IOU_SUCCESS_MAX) and (conf_drop > CONF_DROP_SUCCESS_MIN)

    return {
        "orig": orig_path.name,
        "adv": adv_path.name,
        "iou": iou,
        "conf_orig": conf_orig,
        "conf_adv": conf_adv,
        "conf_drop": conf_drop,
        "n_orig": n_orig,
        "n_adv": n_adv,
        "iou_ok": iou < IOU_SUCCESS_MAX,
        "drop_ok": conf_drop > CONF_DROP_SUCCESS_MIN,
        "success": success,
        "clean": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pairing
# ─────────────────────────────────────────────────────────────────────────────

def discover_pairs(
    orig_dir: Path,
    adv_dir: Path,
) -> list[tuple[Path, Path]]:
    """Pair every adv_NNNN.bin with its original NNNNNN.bin."""
    pattern = re.compile(r"^adv_(\d+)\.bin$")
    pairs: list[tuple[Path, Path]] = []

    for adv_path in sorted(adv_dir.glob("adv_*.bin")):
        m = pattern.match(adv_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        orig = orig_dir / f"{idx:06d}.bin"
        if not orig.exists():
            print(f"  [WARN] No original for {adv_path.name} (expected {orig.name})")
            continue
        pairs.append((orig, adv_path))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict], n_clean: int, n_skipped: int) -> None:
    print("  " + "─" * 78)

    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_iou = sum(1 for r in results if r["iou_ok"])
    n_drop = sum(1 for r in results if r["drop_ok"])

    print(f"  Attacked pairs evaluated : {n}")
    if n_clean:
        print(f"  Cooldown/clean skipped   : {n_clean}  (unattacked copies)")
    if n_skipped:
        print(f"  No-target skipped        : {n_skipped}")

    if n:
        mean_iou = float(np.mean([r["iou"] for r in results]))
        mean_drop = float(np.mean([r["conf_drop"] for r in results]))
        print(f"  IoU  < {IOU_SUCCESS_MAX:<5}           : {n_iou}/{n}  "
              f"(mean IoU {mean_iou:.3f})")
        print(f"  Drop > {CONF_DROP_SUCCESS_MIN:<5}           : {n_drop}/{n}  "
              f"(mean drop {mean_drop:.3f})")
        print(f"  Overall attack success   : {n_success}/{n}  "
              f"({100.0 * n_success / n:.1f}%)  [IoU AND Drop]")
    print("  " + "─" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Module 4 — Attack Success Metrics (IoU + Confidence Drop)"
    )
    parser.add_argument("--seq", type=str, default=DEFAULT_SEQUENCE,
                        help=f"KITTI tracking sequence for the original frames "
                             f"(default: {DEFAULT_SEQUENCE})")
    parser.add_argument("--orig-dir", type=str, default=None,
                        help="Directory with original NNNNNN.bin frames "
                             "(default: the --seq folder under the tracking dataset)")
    parser.add_argument("--adv-dir", type=str, default=str(DEFAULT_ADV_DIR),
                        help="Directory with adversarial adv_NNNN.bin frames")
    parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE,
                        help=f"Voxel size for the detector (default: {VOXEL_SIZE})")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Evaluate at most this many pairs (default: all)")
    parser.add_argument("--include-clean", action="store_true", default=False,
                        help="Also report untouched cooldown frames (IoU=1, drop=0)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    orig_dir = Path(args.orig_dir) if args.orig_dir else default_velodyne_dir(args.seq)
    adv_dir = Path(args.adv_dir)

    print("\n\tModule 4 — Attack Metrics")
    print("-" * 80)

    if not adv_dir.exists():
        print(f"  [ERROR] Adversarial directory not found: {adv_dir}")
        return 1

    pairs = discover_pairs(orig_dir, adv_dir)
    if not pairs:
        print(f"  [ERROR] No adv_*.bin ↔ original pairs found.")
        return 1

    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    print(f"  Original frames    : {orig_dir}")
    print(f"  Adversarial frames : {adv_dir}")
    print(f"  Pairs to evaluate  : {len(pairs)}")
    print(f"  Success criteria   : IoU < {IOU_SUCCESS_MAX} AND "
          f"Drop > {CONF_DROP_SUCCESS_MIN}")
    print("  " + "─" * 78)
    print(f"  {'#':>4} {'Adversarial':<14} {'IoU':>7} {'ConfOrig':>9} "
          f"{'ConfAdv':>8} {'Drop':>7} {'Result':>9}")
    print(f"  {'─' * 78}")

    results: list[dict] = []
    n_clean = 0
    n_skipped = 0
    total = len(pairs)

    for i, (orig_path, adv_path) in enumerate(pairs, 1):
        # Live status so long runs don't look frozen.
        print(f"  {i:>4} {adv_path.name:<14} ...", end="\r", flush=True)

        metrics = evaluate_pair(orig_path, adv_path, voxel_size=args.voxel_size)

        if metrics is None:
            n_skipped += 1
            print(f"  {i:>4} {adv_path.name:<14} "
                  f"{'—':>7} {'—':>9} {'—':>8} {'—':>7} {'no target':>9}")
            continue

        if metrics["clean"]:
            n_clean += 1
            if not args.include_clean:
                print(f"  {i:>4} {adv_path.name:<14} "
                      f"{'1.000':>7} {'—':>9} {'—':>8} {'0.000':>7} "
                      f"{'clean':>9}")
                continue

        co = "—" if metrics["conf_orig"] is None else f"{metrics['conf_orig']:.3f}"
        ca = "—" if metrics["conf_adv"] is None else f"{metrics['conf_adv']:.3f}"
        result = "SUCCESS" if metrics["success"] else "fail"
        iou_flag = "" if metrics["iou_ok"] else "!"
        drop_flag = "" if metrics["drop_ok"] else "!"
        print(f"  {i:>4} {adv_path.name:<14} "
              f"{metrics['iou']:>6.3f}{iou_flag:<1} "
              f"{co:>9} {ca:>8} "
              f"{metrics['conf_drop']:>6.3f}{drop_flag:<1} {result:>9}")

        results.append(metrics)

    if not results:
        print("\n  [ERROR] No evaluable attacked pairs "
              "(only clean frames or no target vehicle).")
        return 1

    _print_summary(results, n_clean, n_skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
attack.py

Module 4 — Attack Module (POPA: Partial Object Persistence)

POPA never lets the target object disappear completely. Instead, *different
parts* of the object vanish and reappear across consecutive frames.

Because a LiDAR only ever captures the surfaces of a vehicle that face the
sensor (the far side/rear are self-occluded), each vehicle is partitioned by
distance from the sensor into range shells ordered nearest → farthest:
    core → f0 → f1 → ...
The nearest "core" shell is the most reliably visible surface and is kept
persistent (the object never fully disappears). The rest of the points are
split into many thin flicker shells; each attack frame keeps the core plus at
most one thin shell, so the surviving object is a small, shifting fragment.
Removal acts only on the real points that exist on the vehicle's surface.

The result:
  * partially preserves the object              (the core is never gone)
  * removes different range shells per frame     (shape/extent keeps changing)
  * destabilises 3-D object detection           (bounding box jumps around)
  * confuses multi-object tracking              (ID switches, track breaks)
  * maintains a realistic LiDAR appearance      (kept shells keep their real
                                                 points; only vehicles are
                                                 touched, never the whole scene)

Attack cadence: the attack runs in *bursts* over the real frame sequence — it
attacks N consecutive frames, then cools down (leaves those frames untouched)
for N frames, then attacks the next N, repeating.
Default cadence is "attack 3, cool down 3":
    attack 000000-000002 → cool 000003-000005 → attack 000006-000008 → ...

Each attacked frame's target object is extracted from *that* frame, and the
adversarial output is named after the source frame number:
    000000.bin → adv_0000.bin,  000006.bin → adv_0006.bin, ...

Inputs (from Module 3):
  1. target_object   : (N, 4)   the extracted object point cloud
  2. regions         : dict     {"front": pts, "side": pts, "rear": pts}
  3. schedule        : dict     {frame_id: [visible_region_names]}
  4. original frame  : (M, 4)   the full raw LiDAR scene

Output:
  Adversarial LiDAR frames written as KITTI .bin files:
      adv_0001.bin, adv_0002.bin, adv_0003.bin, ...
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.voxelization import voxelize
from lidar_pipeline.semantic_segmentation import segment_point_cloud, CLASS_VEHICLE
from lidar_pipeline.instance_segmentation import instance_segment, cluster_summary


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Sensor origin (KITTI Velodyne mounting height)
SENSOR_ORIGIN: np.ndarray = np.array([0.0, 0.0, 1.73], dtype=np.float32)

# Sensor-relative region model (POPA). A vehicle is only ever seen as the
# surfaces facing the LiDAR, so we partition its points by range from the
# sensor into shells ordered nearest → farthest. The nearest "core" shell is
# the most reliably visible part and is kept persistent (the object never fully
# disappears). The rest of the points are divided into many THIN flicker shells;
# each attack frame keeps the core plus at most one thin shell, so the surviving
# object is always a small, shifting fragment (< ~20% of points) — enough to
# persist, small enough to collapse the detector's box and confidence.
CORE_FRACTION: float = 0.10          # nearest fraction always kept
N_FLICKER_SHELLS: int = 10           # thin shells the remaining points split into


def _make_region_names(n_flicker: int) -> tuple[str, ...]:
    """Region names nearest → farthest: ("core", "f0", "f1", ...)."""
    return ("core",) + tuple(f"f{i}" for i in range(n_flicker))


REGION_NAMES: tuple[str, ...] = _make_region_names(N_FLICKER_SHELLS)
PERSISTENT_REGIONS: tuple[str, ...] = ("core",)

# Attack cadence over the real frame sequence: attack ATTACK_BURST consecutive
# frames, then COOLDOWN consecutive frames pass through untouched, repeating.
#   e.g. attack 000000-000002, cool down 000003-000005, attack 000006-000008 ...
ATTACK_BURST: int = 3
COOLDOWN: int = 3

# Padding (metres) around the object bounding box when carving it out of the
# raw scene, so we cleanly remove the *real* object before re-inserting parts.
BBOX_PAD_M: float = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Target-object extraction  (shared helper used across Module 4 submodules)
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_regions(
    obj: np.ndarray,
    axis: str = "x",
    n_regions: int = 3,
    region_names: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Partition an object point cloud into contiguous regions along one axis.

    Points are binned into `n_regions` equal-width slices between the object's
    min and max coordinate on `axis`. Bin 0 (lowest coordinate) maps to the
    first name in `region_names`, and so on.
    """
    if region_names is None:
        region_names = [f"region_{i}" for i in range(n_regions)]
    n_regions = len(region_names)

    if axis not in AXIS_INDEX:
        raise ValueError(f"axis must be one of {list(AXIS_INDEX)}, got {axis!r}")
    ax = AXIS_INDEX[axis]

    coords = obj[:, ax].astype(np.float64)
    lo, hi = float(coords.min()), float(coords.max())
    span = max(hi - lo, 1e-8)

    bin_ids = np.clip(
        ((coords - lo) / span * n_regions).astype(int),
        0, n_regions - 1,
    )

    regions: dict[str, np.ndarray] = {}
    for i, name in enumerate(region_names):
        regions[name] = obj[bin_ids == i].astype(np.float32)
    return regions


def extract_target_and_regions(
    frame: np.ndarray,
    voxel_size: float = 0.1,
    axis: str = "x",
    n_regions: int = 3,
    region_names: Optional[list[str]] = None,
    pick_largest: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """
    Detect a vehicle object in a raw LiDAR frame and split it into regions.

    Pipeline: voxelize → semantic segmentation → instance segmentation →
    pick a vehicle instance → partition into `region_names` along `axis`.

    Inputs:
        frame:        (M, 4) raw LiDAR point cloud [x, y, z, intensity]
        voxel_size:   voxel edge length (m) used for scene pre-processing
        axis:         axis to slice the object along ("x", "y", or "z")
        n_regions:    number of regions (overridden by len(region_names))
        region_names: names for the slices, ordered low→high coordinate
        pick_largest: if True pick the vehicle with the most points,
                      otherwise the lowest instance id

    Returns:
        target_object : (N, 4) point cloud of the chosen vehicle
        regions       : dict {region_name: (k, 4) points}
        info          : cluster-summary dict for the chosen instance

    Raises:
        RuntimeError: if no vehicle instance is found in the frame.
    """
    if frame.ndim != 2 or frame.shape[1] != 4:
        raise ValueError(f"frame must be (M, 4), got {frame.shape}")

    if region_names is None:
        region_names = ["rear", "side", "front"][:n_regions]

    vox = voxelize(frame, voxel_size=voxel_size)
    sem_labels = segment_point_cloud(vox)
    inst_ids = instance_segment(vox, sem_labels)

    rows = cluster_summary(sem_labels, inst_ids)
    vehicle_rows = [r for r in rows if r["semantic_class"] == CLASS_VEHICLE]

    if not vehicle_rows:
        raise RuntimeError("No vehicle object detected in this frame.")

    if pick_largest:
        target_row = max(vehicle_rows, key=lambda r: r["n_points"])
    else:
        target_row = min(vehicle_rows, key=lambda r: r["instance_id"])

    mask = inst_ids == target_row["instance_id"]
    target_object = vox[mask].astype(np.float32)

    regions = _split_into_regions(
        target_object, axis=axis, n_regions=n_regions, region_names=region_names,
    )
    return target_object, regions, target_row


def _split_by_sensor_range(
    obj: np.ndarray,
    sensor_origin: np.ndarray,
    region_names: tuple[str, ...],
    core_fraction: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Partition a vehicle's points into range shells relative to the sensor.

    Shells are ordered nearest → farthest. The first ("core") shell holds the
    nearest `core_fraction` of the points (the most reliably visible surface);
    the remaining points are split into equal-count shells for the rest of
    `region_names`.

    Returns (regions_dict, range_edges) where range_edges has len(region_names)+1
    entries defining each shell's [lo, hi) distance from the sensor.
    """
    n = len(region_names)
    r = np.linalg.norm(obj[:, :3].astype(np.float64) - sensor_origin, axis=1)

    # Cumulative point-fraction breakpoints: core_fraction, then even splits.
    rest = max(1.0 - core_fraction, 0.0)
    quantiles = [0.0, core_fraction]
    for k in range(1, n):
        quantiles.append(core_fraction + rest * k / (n - 1))
    quantiles[-1] = 1.0

    edges = np.quantile(r, quantiles) if len(r) > 0 else np.zeros(n + 1)
    # Make outer bound inclusive of the farthest point.
    edges[-1] = edges[-1] + 1e-6

    regions: dict[str, np.ndarray] = {}
    for i, name in enumerate(region_names):
        lo, hi = edges[i], edges[i + 1]
        if i == n - 1:
            mask = r >= lo
        else:
            mask = (r >= lo) & (r < hi)
        regions[name] = obj[mask].astype(np.float32)

    return regions, edges.astype(np.float64)


def extract_all_vehicles(
    frame: np.ndarray,
    voxel_size: float = 0.1,
    region_names: tuple[str, ...] = REGION_NAMES,
    core_fraction: float = CORE_FRACTION,
    sensor_origin: Optional[np.ndarray] = None,
    min_points: int = 5,
) -> list[dict]:
    """
    Detect *every* vehicle instance in a frame and describe each for POPA.

    Each vehicle is partitioned into sensor-relative range shells (nearest →
    farthest). The attack operates only on these vehicle instances — all other
    points in the scene (ground, buildings, vegetation, ...) are left untouched.

    Returns a list of dicts, one per vehicle:
        {
          "object":      (N, 4) voxelized points of the vehicle,
          "bbox_min":    (3,) lower corner,   "bbox_max": (3,) upper corner,
          "regions":     {region_name: (k, 4) points},  nearest → farthest,
          "range_edges": (len(region_names)+1,) shell distance breakpoints,
          "info":        cluster-summary dict,
        }
    """
    if frame.ndim != 2 or frame.shape[1] != 4:
        raise ValueError(f"frame must be (M, 4), got {frame.shape}")

    origin = (sensor_origin if sensor_origin is not None
              else SENSOR_ORIGIN).astype(np.float64)

    vox = voxelize(frame, voxel_size=voxel_size)
    sem_labels = segment_point_cloud(vox)
    inst_ids = instance_segment(vox, sem_labels)
    rows = cluster_summary(sem_labels, inst_ids)

    vehicles: list[dict] = []
    for row in rows:
        if row["semantic_class"] != CLASS_VEHICLE:
            continue
        if row["n_points"] < min_points:
            continue

        mask = inst_ids == row["instance_id"]
        obj = vox[mask].astype(np.float32)
        regions, edges = _split_by_sensor_range(
            obj, origin, region_names, core_fraction,
        )
        vehicles.append({
            "object": obj,
            "bbox_min": obj[:, :3].min(axis=0),
            "bbox_max": obj[:, :3].max(axis=0),
            "regions": regions,
            "range_edges": edges,
            "info": row,
        })

    return vehicles


# ─────────────────────────────────────────────────────────────────────────────
# POPA temporal schedule
# ─────────────────────────────────────────────────────────────────────────────

def _dedupe(seq: list[str]) -> list[str]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _region_subset_patterns(non_persistent: list[str]) -> list[list[str]]:
    """
    Cycle of visibility patterns for the flickering (non-persistent) shells.

    For non_persistent = ["mid", "far"] this yields:
        [], ["mid"], ["far"]
    i.e. the empty set plus each single shell — so the object cycles through
    core-only, core+mid, core+far, core-only, ... across the attack burst.
    """
    patterns: list[list[str]] = [[]]
    patterns.extend([[r] for r in non_persistent])
    return patterns


def is_attack_frame(
    frame_index: int,
    attack_burst: int = ATTACK_BURST,
    cooldown: int = COOLDOWN,
) -> bool:
    """
    True if a real dataset frame (0-based index) falls in an attack burst.

    With attack_burst=3, cooldown=3: indices 0,1,2 attack; 3,4,5 cool down;
    6,7,8 attack; and so on.
    """
    period = attack_burst + cooldown
    return (frame_index % period) < attack_burst


def popa_regions_for_frame(
    frame_index: int,
    region_names: tuple[str, ...] = REGION_NAMES,
    persistent: tuple[str, ...] = PERSISTENT_REGIONS,
    attack_burst: int = ATTACK_BURST,
    cooldown: int = COOLDOWN,
) -> list[str]:
    """
    Visible shells for a given real frame index under the POPA cadence.

    Attack frames keep the persistent shell(s) plus a flickering subset that
    depends on the frame's position within the burst, so different parts vanish
    across the burst (core-only, core+mid, core+far, ...). Cooldown frames
    return the full object.
    """
    period = attack_burst + cooldown
    pos = frame_index % period

    if pos >= attack_burst:
        # Cooldown frame: whole object visible.
        return list(region_names)

    non_persistent = [r for r in region_names if r not in persistent]
    patterns = _region_subset_patterns(non_persistent)
    extra = patterns[pos % len(patterns)]
    return _dedupe(list(persistent) + list(extra))


def build_popa_schedule(
    frame_indices: list[int],
    region_names: tuple[str, ...] = REGION_NAMES,
    persistent: tuple[str, ...] = PERSISTENT_REGIONS,
    attack_burst: int = ATTACK_BURST,
    cooldown: int = COOLDOWN,
) -> dict[int, dict]:
    """
    Build a POPA schedule over real dataset frame indices.

    Attacks `attack_burst` consecutive frames, then lets `cooldown` consecutive
    frames pass through untouched, repeating across the whole sequence.

    Returns:
        {frame_index: {"mode": "attack"|"cooldown", "kept": [region_names]}}
    """
    if attack_burst < 1 or cooldown < 0:
        raise ValueError("attack_burst must be >=1 and cooldown >=0")

    schedule: dict[int, dict] = {}
    for idx in frame_indices:
        attack = is_attack_frame(idx, attack_burst, cooldown)
        kept = popa_regions_for_frame(
            idx, region_names, persistent, attack_burst, cooldown,
        )
        schedule[idx] = {"mode": "attack" if attack else "cooldown", "kept": kept}
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# POPA per-frame synthesis  (vehicle-only, region-localized removal)
# ─────────────────────────────────────────────────────────────────────────────

def _vehicle_bbox_mask(
    raw_xyz: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    pad_m: float,
) -> np.ndarray:
    """Boolean mask of raw points inside a vehicle's (padded) bounding box."""
    inside = np.ones(len(raw_xyz), dtype=bool)
    for a in range(3):
        inside &= (raw_xyz[:, a] >= bbox_min[a] - pad_m) & \
                  (raw_xyz[:, a] <= bbox_max[a] + pad_m)
    return inside


def popa_attack_frame(
    original_frame: np.ndarray,
    vehicles: list[dict],
    kept_regions: list[str],
    region_names: tuple[str, ...] = REGION_NAMES,
    sensor_origin: Optional[np.ndarray] = None,
    pad_m: float = BBOX_PAD_M,
) -> np.ndarray:
    """
    Produce one adversarial LiDAR frame under POPA — attacking vehicles only.

    For every vehicle, the sensor-range shells that are *not* in `kept_regions`
    have their real LiDAR points removed; the kept shells retain their full
    original points, and every non-vehicle point in the scene is preserved.

    A raw point is removed only if it falls inside the vehicle's bounding box
    AND its distance from the sensor lies in a missing shell — so removal is
    confined to the vehicle's actually-visible surface returns.
    """
    if original_frame.ndim != 2 or original_frame.shape[1] != 4:
        raise ValueError(f"original_frame must be (M, 4), got {original_frame.shape}")

    origin = (sensor_origin if sensor_origin is not None
              else SENSOR_ORIGIN).astype(np.float64)

    raw_xyz = original_frame[:, :3].astype(np.float64)
    r_all = np.linalg.norm(raw_xyz - origin, axis=1)   # range of every point
    remove = np.zeros(len(original_frame), dtype=bool)

    n = len(region_names)
    for v in vehicles:
        missing = [r for r in region_names if r not in kept_regions]
        if not missing:
            continue

        inbox = _vehicle_bbox_mask(raw_xyz, v["bbox_min"], v["bbox_max"], pad_m)
        if not inbox.any():
            continue

        edges = v["range_edges"]
        for name in missing:
            i = region_names.index(name)
            lo, hi = edges[i], edges[i + 1]
            if i == n - 1:
                band = r_all >= lo
            else:
                band = (r_all >= lo) & (r_all < hi)
            remove |= inbox & band

    return original_frame[~remove].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Driver — attack a real frame sequence with the burst/cooldown cadence
# ─────────────────────────────────────────────────────────────────────────────

def _adv_name(frame_stem: str, prefix: str) -> str:
    """
    Map a source frame filename stem to its adversarial output name, keeping the
    frame number so outputs line up with originals.
        "000000" -> "adv_0000.bin",  "000006" -> "adv_0006.bin"
    Falls back to the raw stem if it is not purely numeric.
    """
    try:
        return f"{prefix}{int(frame_stem):04d}.bin"
    except ValueError:
        return f"{prefix}{frame_stem}.bin"


def generate_adversarial_sequence(
    frame_paths: list[Path],
    output_dir: str | Path,
    attack_burst: int = ATTACK_BURST,
    cooldown: int = COOLDOWN,
    voxel_size: float = 0.1,
    region_names: tuple[str, ...] = REGION_NAMES,
    persistent: tuple[str, ...] = PERSISTENT_REGIONS,
    core_fraction: float = CORE_FRACTION,
    sensor_origin: Optional[np.ndarray] = None,
    prefix: str = "adv_",
    pad_m: float = BBOX_PAD_M,
    min_points: int = 5,
    write_cooldown: bool = True,
    random_seed: Optional[int] = 42,
    verbose: bool = False,
) -> list[Path]:
    """
    Run POPA over an ordered list of real LiDAR frames.

    For each frame, its 0-based position in `frame_paths` decides the cadence:
    attack bursts get POPA (target extracted from that very frame; regions
    flicker), cooldown frames pass through untouched. Outputs are named after
    the source frame number, e.g. 000006.bin -> adv_0006.bin.

    Cooldown frames are written as clean copies when `write_cooldown` is True so
    the output directory is a complete, continuous sequence; set it False to
    write only the attacked frames.

    Returns the list of written file paths.
    """
    from lidar_pipeline.loader import load_bin_file

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    reports: list[dict] = []

    for idx, fpath in enumerate(frame_paths):
        raw = load_bin_file(fpath)
        attack = is_attack_frame(idx, attack_burst, cooldown)
        out_name = _adv_name(fpath.stem, prefix)

        if not attack:
            # Cooldown: leave the whole frame untouched.
            report = {
                "idx": idx, "src": fpath.name, "out": out_name,
                "mode": "cooldown", "kept": list(region_names),
                "n_vehicles": None, "n_removed": 0,
                "n_scene": len(raw), "written": False,
            }
            if write_cooldown:
                raw.astype(np.float32).tofile(out_dir / out_name)
                saved.append(out_dir / out_name)
                report["written"] = True
            reports.append(report)
            continue

        # Attack: detect every vehicle, partition into sensor-range shells, and
        # remove the missing shells. Non-vehicle points are never touched.
        vehicles = extract_all_vehicles(
            frame=raw, voxel_size=voxel_size, region_names=region_names,
            core_fraction=core_fraction, sensor_origin=sensor_origin,
            min_points=min_points,
        )

        if not vehicles:
            # No vehicle to attack → clean pass-through.
            raw.astype(np.float32).tofile(out_dir / out_name)
            saved.append(out_dir / out_name)
            reports.append({
                "idx": idx, "src": fpath.name, "out": out_name,
                "mode": "attack (no vehicle)", "kept": [],
                "n_vehicles": 0, "n_removed": 0,
                "n_scene": len(raw), "written": True,
            })
            continue

        kept = popa_regions_for_frame(
            idx, region_names, persistent, attack_burst, cooldown,
        )

        adv = popa_attack_frame(
            original_frame=raw, vehicles=vehicles, kept_regions=kept,
            region_names=region_names, sensor_origin=sensor_origin, pad_m=pad_m,
        )
        adv.astype(np.float32).tofile(out_dir / out_name)
        saved.append(out_dir / out_name)

        reports.append({
            "idx": idx, "src": fpath.name, "out": out_name, "mode": "attack",
            "kept": kept, "n_vehicles": len(vehicles),
            "n_removed": len(raw) - len(adv), "n_scene": len(adv),
            "written": True,
        })

    if verbose:
        _print_popa_report(reports, region_names, out_dir, write_cooldown)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_popa_report(
    reports: list[dict],
    region_names: tuple[str, ...],
    out_dir: Path,
    write_cooldown: bool,
) -> None:
    print(f"\n  POPA Attack (Module 4) — vehicles only")
    print("  " + "─" * 74)
    print(f"  Output directory : {out_dir}")
    print(f"  Cooldown frames  : {'written (clean copies)' if write_cooldown else 'skipped'}")
    print("  " + "─" * 74)

    print(f"  {'Src':<12} {'Out':<14} {'Mode':<20} {'Visible':<18} "
          f"{'Veh':>4} {'Removed':>8}")
    print(f"  {'─' * 74}")

    for r in reports:
        visible = ", ".join(r["kept"]) if r["kept"] else "-"
        missing = [k for k in region_names if k not in r["kept"]]
        if r["mode"] == "attack" and missing:
            visible += f" (−{len(missing)} shells)"
        veh = "" if r["n_vehicles"] is None else str(r["n_vehicles"])
        removed = f"{r['n_removed']:,}" if r["n_removed"] else ""
        skip = "" if r["written"] else "  [skipped]"
        print(f"  {r['src']:<12} {r['out']:<14} {r['mode']:<20} "
              f"{visible:<18} {veh:>4} {removed:>8}{skip}")

    print("  " + "─" * 74)
    n_attack = sum(1 for r in reports if r["mode"].startswith("attack"))
    n_written = sum(1 for r in reports if r["written"])
    total_removed = sum(r["n_removed"] for r in reports)
    print(f"  Frames processed : {len(reports)}  "
          f"(attack={n_attack}, cooldown={len(reports) - n_attack})")
    print(f"  Vehicle points removed (total) : {total_removed:,}")
    print(f"  Files written    : {n_written}\n")



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from lidar_pipeline.loader import (
        get_all_frames, load_bin_file, default_velodyne_dir, DEFAULT_SEQUENCE,
    )

    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Module 4 — POPA Attack (Partial Object Persistence)"
    )
    parser.add_argument(
        "--seq", type=str, default=DEFAULT_SEQUENCE,
        help=f"KITTI tracking sequence id to attack (default: {DEFAULT_SEQUENCE})",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override the source velodyne folder (default: the --seq folder "
             "under the tracking dataset)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="0-based index of the first dataset frame to process (default: 0)",
    )
    parser.add_argument(
        "--count", type=int, default=12,
        help="Number of consecutive dataset frames to process (default: 12)",
    )
    parser.add_argument(
        "--attack-burst", type=int, default=ATTACK_BURST,
        help=f"Consecutive frames attacked per cycle (default: {ATTACK_BURST})",
    )
    parser.add_argument(
        "--cooldown", type=int, default=COOLDOWN,
        help=f"Consecutive cooldown frames per cycle (default: {COOLDOWN})",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.1,
        help="Voxel size in metres for scene pre-processing (default: 0.1)",
    )
    parser.add_argument(
        "--core-fraction", type=float, default=CORE_FRACTION,
        help=f"Nearest fraction of points kept as the persistent visible core "
             f"(default: {CORE_FRACTION})",
    )
    parser.add_argument(
        "--n-flicker", type=int, default=N_FLICKER_SHELLS,
        help=f"Number of thin flicker shells the rest of the points split into "
             f"(default: {N_FLICKER_SHELLS}); more shells = each kept fragment "
             f"is smaller = larger confidence drop",
    )
    parser.add_argument(
        "--skip-cooldown", action="store_true", default=False,
        help="Only write attacked frames (skip clean cooldown copies)",
    )
    parser.add_argument(
        "--out", type=str, default="data/adversarial",
        help="Output directory for adv_XXXX.bin (default: data/adversarial)",
    )
    args = parser.parse_args()

    print("\n\tModule 4 — POPA Attack")
    print("-" * 72)

    DATA_DIR = Path(args.data_dir) if args.data_dir else default_velodyne_dir(args.seq)

    all_frames = get_all_frames(DATA_DIR)
    if not (0 <= args.start < len(all_frames)):
        print(f"\n  [ERROR] Start index {args.start} out of range "
              f"(0–{len(all_frames) - 1}).")
        sys.exit(1)

    end = min(args.start + args.count, len(all_frames))
    frame_paths = all_frames[args.start:end]
    print(f"\n  Dataset frames : {frame_paths[0].name} … {frame_paths[-1].name} "
          f"({len(frame_paths)} frames)")
    print(f"  Cadence        : attack {args.attack_burst}, "
          f"cool down {args.cooldown}")

    region_names = _make_region_names(args.n_flicker)

    saved = generate_adversarial_sequence(
        frame_paths=frame_paths,
        output_dir=args.out,
        attack_burst=args.attack_burst,
        cooldown=args.cooldown,
        voxel_size=args.voxel_size,
        region_names=region_names,
        persistent=PERSISTENT_REGIONS,
        core_fraction=args.core_fraction,
        write_cooldown=not args.skip_cooldown,
        random_seed=42,
        verbose=True,
    )

    # POPA guarantee: every attacked frame keeps the persistent visible core.
    for idx in range(len(frame_paths)):
        if is_attack_frame(idx, args.attack_burst, args.cooldown):
            kept = popa_regions_for_frame(
                idx, region_names, PERSISTENT_REGIONS,
                args.attack_burst, args.cooldown,
            )
            for pr in PERSISTENT_REGIONS:
                assert pr in kept, f"Frame {idx}: '{pr}' must persist"

    print(f"  Example outputs:")
    for p in saved[:4]:
        arr = load_bin_file(p)
        print(f"    {p.name:<16}  shape {arr.shape}")
    if len(saved) > 4:
        print(f"    ... and {len(saved) - 4} more")
    print()

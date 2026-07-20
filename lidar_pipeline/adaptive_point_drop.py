# adaptive_point_drop.py — Submodule 4.1.1: Adaptive Point Dropping
# Remove strategically selected points from a target object while preserving overall object visibility.

from __future__ import annotations

import numpy as np
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_SURVIVAL_FRACTION: float = 0.0
PROTECTED_REGION: str = "front"

# front always 0%, non-front maximally aggressive to collapse the adversarial bbox
DEFAULT_DROP_RATIOS: dict[str, float] = {
    "front": 0.00,
    "side":  0.995,
    "rear":  1.00,
}

# temporal schedule: region-aware removal cycling across frames
# Every frame aggressively drops both non-anchor regions.
# Near-total removal to ensure adversarial bbox shrinks well below IoU 0.65
# Frame 1 → total rear removal, near-total side
# Frame 2 → near-total rear, total side removal
# Frame 3 → total removal of both rear + side
DEFAULT_TEMPORAL_SCHEDULE: dict[int, dict[str, float]] = {
    1: {"front": 0.00, "side": 0.995, "rear": 1.00},
    2: {"front": 0.00, "side": 1.00,  "rear": 0.995},
    3: {"front": 0.00, "side": 1.00,  "rear": 1.00},
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate_inputs(
    target_object: np.ndarray,
    regions: dict[str, np.ndarray],
) -> None:
    # basic shape checks on the target object and each region
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


def _resolve_drop_ratios(
    regions: dict[str, np.ndarray],
    frame_id: int,
    drop_ratios: Optional[dict[str, float]],
    temporal_schedule: Optional[dict[int, dict[str, float]]],
    protected_region: str,
) -> tuple[dict[str, float], str]:
    # pick the ratio table: explicit override > custom schedule > default schedule > defaults
    if drop_ratios is not None:
        base = drop_ratios
        source = "custom drop_ratios"
    elif temporal_schedule is not None:
        # Cycle frame_id into the schedule's key range
        schedule_keys = sorted(temporal_schedule.keys())
        n_entries = len(schedule_keys)
        cycled_key = schedule_keys[(frame_id - 1) % n_entries] if n_entries > 0 else None
        if cycled_key is not None and cycled_key in temporal_schedule:
            base = temporal_schedule[cycled_key]
            source = f"custom temporal_schedule[{cycled_key}] (frame_id={frame_id})"
        else:
            base = DEFAULT_DROP_RATIOS
            source = "DEFAULT_DROP_RATIOS (schedule fallback)"
    else:
        # Cycle frame_id into the default schedule's key range
        schedule_keys = sorted(DEFAULT_TEMPORAL_SCHEDULE.keys())
        n_entries = len(schedule_keys)
        cycled_key = schedule_keys[(frame_id - 1) % n_entries] if n_entries > 0 else None
        if cycled_key is not None and cycled_key in DEFAULT_TEMPORAL_SCHEDULE:
            base = DEFAULT_TEMPORAL_SCHEDULE[cycled_key]
            source = f"DEFAULT_TEMPORAL_SCHEDULE[{cycled_key}] (frame_id={frame_id})"
        else:
            base = DEFAULT_DROP_RATIOS
            source = "DEFAULT_DROP_RATIOS"

    resolved: dict[str, float] = {}
    for name in regions:
        ratio = float(base.get(name, 0.0))
        resolved[name] = max(0.0, min(1.0, ratio))

    # front is always protected — force 0% drop
    if protected_region in resolved:
        resolved[protected_region] = 0.0

    return resolved, source


def _drop_region_points(
    pts: np.ndarray,
    drop_ratio: float,
    rng: np.random.Generator,
    collapse_target: Optional[np.ndarray] = None,
) -> np.ndarray:
    # structured mode: drop points farthest from the collapse target first — collapses bbox
    # When collapse_target is the front centroid, surviving points cluster near the front,
    # shrinking the adversarial bbox away from the original rear/side boundaries.
    # random mode: uniform random dropout (used for front region where ratio=0 anyway)
    n = len(pts)
    if n == 0 or drop_ratio <= 0.0:
        return pts.copy()
    if drop_ratio >= 1.0:
        return np.empty((0, 4), dtype=np.float32)

    n_keep = n - int(round(n * drop_ratio))
    if n_keep <= 0:
        return np.empty((0, 4), dtype=np.float32)

    if collapse_target is not None:
        # structured: sort by distance from collapse target, keep the closest n_keep
        dists = np.linalg.norm(pts[:, :3] - collapse_target, axis=1)
        sorted_idx = np.argsort(dists)      # ascending — closest first
        keep_idx = sorted_idx[:n_keep]
    else:
        keep_idx = rng.choice(n, size=n_keep, replace=False)

    keep_idx.sort()
    return pts[keep_idx].astype(np.float32)


def _enforce_minimum_survival(
    survived_parts: dict[str, np.ndarray],
    original_object: np.ndarray,
    min_survival_frac: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    # if too many points were dropped, inject random originals to meet the 30% floor
    min_required = int(np.ceil(len(original_object) * min_survival_frac))
    current_total = sum(len(v) for v in survived_parts.values())

    if current_total >= min_required:
        return survived_parts

    shortfall = min_required - current_total
    inject_idx = rng.choice(len(original_object), size=shortfall, replace=False)
    extra_pts = original_object[inject_idx].astype(np.float32)

    survived_parts["_injected"] = extra_pts
    return survived_parts


# ── Public API ─────────────────────────────────────────────────────────────────

def adaptive_point_drop(
    target_object: np.ndarray,
    regions: dict[str, np.ndarray],
    frame_id: int = 1,
    frame_name: Optional[str] = None,
    drop_ratios: Optional[dict[str, float]] = None,
    temporal_schedule: Optional[dict[int, dict[str, float]]] = None,
    protected_region: str = PROTECTED_REGION,
    min_survival_frac: float = MIN_SURVIVAL_FRACTION,
    collapse_target: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:

    _validate_inputs(target_object, regions)
    rng = np.random.default_rng(random_seed)

    effective_ratios, schedule_source = _resolve_drop_ratios(
        regions=regions,
        frame_id=frame_id,
        drop_ratios=drop_ratios,
        temporal_schedule=temporal_schedule,
        protected_region=protected_region,
    )

    # Use the provided collapse target (typically the front/anchor centroid)
    # so surviving points cluster near the anchor, collapsing the bbox.
    # Fall back to object centroid if no collapse target is given.
    ct = collapse_target
    if ct is None:
        ct = target_object[:, :3].mean(axis=0).astype(np.float32)

    survived_parts: dict[str, np.ndarray] = {}
    for name, pts in regions.items():
        ratio = effective_ratios.get(name, 0.0)
        # front uses random (ratio=0), non-front uses structured boundary-aware dropout
        target = None if name == protected_region else ct
        survived_parts[name] = _drop_region_points(pts, ratio, rng, target)

    # Only enforce minimum survival if fraction is positive
    if min_survival_frac > 0.0:
        survived_parts = _enforce_minimum_survival(
            survived_parts=survived_parts,
            original_object=target_object,
            min_survival_frac=min_survival_frac,
            rng=rng,
        )

    # assemble — protected region first, then the rest in dict order
    ordered_keys = (
        [protected_region] if protected_region in survived_parts else []
    ) + [k for k in survived_parts if k != protected_region]

    parts = [survived_parts[k] for k in ordered_keys if len(survived_parts[k]) > 0]

    if not parts:
        # With aggressive drop ratios (e.g. 1.0), all non-anchor points
        # may be legitimately removed.  Return empty — caller handles this.
        return np.empty((0, 4), dtype=np.float32)

    attacked_object = np.concatenate(parts, axis=0).astype(np.float32)

    if verbose:
        _print_report(regions, survived_parts, effective_ratios,
                      target_object, attacked_object, frame_id,
                      frame_name, schedule_source, protected_region,
                      min_survival_frac)

    return attacked_object


# ── Verbose reporting ──────────────────────────────────────────────────────────

def _print_report(
    regions: dict[str, np.ndarray],
    survived_parts: dict[str, np.ndarray],
    effective_ratios: dict[str, float],
    target_object: np.ndarray,
    attacked_object: np.ndarray,
    frame_id: int,
    frame_name: Optional[str],
    schedule_source: str,
    protected_region: str,
    min_survival_frac: float,
) -> None:
    n_orig = len(target_object)
    n_out  = len(attacked_object)
    surv_pct = 100.0 * n_out / max(n_orig, 1)

    label = f"{frame_id} ({frame_name})" if frame_name else str(frame_id)
    print(f"\n  Adaptive Point Drop")
    print(f"  {'Region':<12} {'Original':>10} {'Survived':>10} {'Dropped':>10} {'Drop %':>8}")
    print(f"  {'-'*56}")

    all_keys = list(regions.keys()) + [k for k in survived_parts if k not in regions]
    for name in all_keys:
        orig_n = len(regions.get(name, []))
        surv_n = len(survived_parts.get(name, []))
        drop_n = orig_n - min(surv_n, orig_n)
        ratio_pct = effective_ratios.get(name, 0.0) * 100
        tag = "  <- protected" if name == protected_region else ""
        print(f"  {name:<12} {orig_n:>10,} {surv_n:>10,} {drop_n:>10,} {ratio_pct:>7.1f}%{tag}")

    print(f"  {'-'*56}")
    print(f"  {'TOTAL':<12} {n_orig:>10,} {n_out:>10,}")
    ok = n_out >= int(np.ceil(n_orig * min_survival_frac))
    print(f"\n  Survival: {surv_pct:.1f}%  "
          f"{'OK' if ok else 'FAIL'}")
    print(f"  {'-'*56}\n")


if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lidar_pipeline.loader import load_bin_file, get_all_frames, default_velodyne_dir
    from lidar_pipeline.attack import extract_target_and_regions

    sys.stdout.reconfigure(encoding="utf-8")

    DATA_DIR = default_velodyne_dir()

    parser = argparse.ArgumentParser(description="Submodule 4.1 — Adaptive Point Drop")
    parser.add_argument("--frame", type=int, required=True,
                        help="0-based frame index to load from the dataset")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="Voxel size in metres for scene pre-processing (default: 0.1)")
    args = parser.parse_args()

    print("\n\tSubmodule 4.1 — Adaptive Point Drop")
    print("-" * 60)

    frames = get_all_frames(DATA_DIR)
    if not (0 <= args.frame < len(frames)):
        print(f"\n  [ERROR] Frame index {args.frame} out of range (0–{len(frames)-1}).")
        sys.exit(1)

    frame_path = frames[args.frame]
    raw = load_bin_file(frame_path)

    print(f"\n Frame: {frame_path.name}")
    print(f"  Raw scene shape : {raw.shape}")

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

    attacked = adaptive_point_drop(
        target_object=target_object,
        regions=regions,
        frame_id=args.frame,
        frame_name=frame_path.name,
        verbose=True,
        random_seed=42,
    )

    print(f"  Output — Attacked object shape : {attacked.shape}")
    assert attacked.shape[1] == 4, "Output must have 4 columns"
    assert len(attacked) >= int(0.30 * N), "Must retain >= 30% of original points"

"""
instance_segmentation.py

Separate individual objects within each semantic class using the
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
algorithm (Ester et al., KDD 1996).
"""

import sys
import numpy as np
from pathlib import Path
from typing import NamedTuple

# Ensure the project root is on sys.path so `lidar_pipeline` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import load_bin_file, get_all_frames, default_velodyne_dir
from lidar_pipeline.voxelization import voxelize

from lidar_pipeline.semantic_segmentation import (
    segment_point_cloud,
    CLASS_UNLABELLED,
    CLASS_GROUND,
    CLASS_VEGETATION,
    CLASS_BUILDING,
    CLASS_VEHICLE,
    CLASS_PERSON,
    CLASS_NAMES,
)


# DBSCAN parameters per semantic class

class DbscanParams(NamedTuple):
    eps:         float
    min_samples: int


# Classes that do NOT get instance-level clustering
SKIP_CLASSES = {CLASS_UNLABELLED, CLASS_GROUND}

# Per-class tuned DBSCAN hyper-parameters
CLASS_DBSCAN_PARAMS: dict[int, DbscanParams] = {
    CLASS_VEHICLE    : DbscanParams(eps=1.0, min_samples=5),
    CLASS_PERSON     : DbscanParams(eps=0.4, min_samples=3),
    CLASS_VEGETATION : DbscanParams(eps=0.6, min_samples=4),
    CLASS_BUILDING   : DbscanParams(eps=1.5, min_samples=10),
}

# Fallback for any class not explicitly listed above
DEFAULT_DBSCAN_PARAMS = DbscanParams(eps=0.8, min_samples=4)

NOISE_ID = -1


# Core DBSCAN

def _dbscan(
    xyz:         np.ndarray,
    eps:         float,
    min_samples: int,
) -> np.ndarray:

    M = len(xyz)
    if M == 0:
        return np.empty(0, dtype=np.int32)

    labels     = np.full(M, NOISE_ID, dtype=np.int32)
    visited    = np.zeros(M, dtype=bool)
    cluster_id = 0
    eps2       = eps * eps


    def _region_query(p_idx: int) -> np.ndarray:
        # Return indices of all points within ε of xyz[p_idx]
        diff  = xyz - xyz[p_idx]              
        dist2 = (diff * diff).sum(axis=1)     
        return np.where(dist2 <= eps2)[0]    

    for p in range(M):
        if visited[p]:
            continue

        visited[p]   = True
        neighbours   = _region_query(p)

        if len(neighbours) < min_samples:
            # p is noise (may be absorbed later as a border point)
            labels[p] = NOISE_ID
            continue

        # Start a new cluster
        labels[p] = cluster_id
        seed_set   = list(neighbours)

        idx = 0
        while idx < len(seed_set):
            q = seed_set[idx]
            idx += 1

            if not visited[q]:
                visited[q] = True
                q_neighbours = _region_query(q)

                if len(q_neighbours) >= min_samples:
                    # q is a core point → expand its neighbours into the cluster
                    seed_set.extend(q_neighbours.tolist())

            # Absorb q into the current cluster (even if it was noise before)
            if labels[q] == NOISE_ID:
                labels[q] = cluster_id

        cluster_id += 1

    return labels


# Per-class instance segmentation

def instance_segment_class(
    xyz:          np.ndarray,
    semantic_cls: int,
) -> np.ndarray:

    params = CLASS_DBSCAN_PARAMS.get(semantic_cls, DEFAULT_DBSCAN_PARAMS)
    return _dbscan(xyz, eps=params.eps, min_samples=params.min_samples)


# Full instance segmentation

def instance_segment(
    points:         np.ndarray,
    semantic_labels: np.ndarray,
) -> np.ndarray:

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) array, got shape {points.shape}")
    if len(semantic_labels) != len(points):
        raise ValueError(
            f"points ({len(points)}) and semantic_labels ({len(semantic_labels)}) "
            "must have the same length."
        )

    N             = len(points)
    instance_ids  = np.full(N, NOISE_ID, dtype=np.int32)
    global_offset = 0

    xyz = points[:, :3]

    for cls_id in CLASS_NAMES:
        if cls_id in SKIP_CLASSES:
            continue

        mask = semantic_labels == cls_id
        if not np.any(mask):
            continue

        class_xyz    = xyz[mask]
        local_ids    = instance_segment_class(class_xyz, cls_id)

        positive = local_ids >= 0
        shifted  = np.where(positive, local_ids + global_offset, NOISE_ID)

        instance_ids[mask] = shifted

        # Advance offset past all cluster IDs used by this class
        n_clusters    = int(local_ids.max()) + 1 if positive.any() else 0
        global_offset += n_clusters

    return instance_ids


# Cluster summary

def cluster_summary(
    semantic_labels: np.ndarray,
    instance_ids:    np.ndarray,
) -> list[dict]:

    rows = []
    unique_inst = np.unique(instance_ids)

    for inst in unique_inst:
        if inst == NOISE_ID:
            continue
        mask    = instance_ids == inst
        cls_id  = int(np.bincount(semantic_labels[mask]).argmax())
        rows.append({
            "instance_id":    int(inst),
            "semantic_class": cls_id,
            "class_name":     CLASS_NAMES.get(cls_id, "unknown"),
            "n_points":       int(mask.sum()),
        })

    # Sort by instance_id for deterministic output
    rows.sort(key=lambda r: r["instance_id"])
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    DATA_DIR = default_velodyne_dir()

    print("\t Instance Segmentation  (DBSCAN)")
    print("-" * 60)

    # Load & pre-process
    try:
        frames = get_all_frames(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    fp  = frames[0]
    print(f"\nFrame : {fp.name}")

    raw = load_bin_file(fp)
    vox = voxelize(raw, voxel_size=0.1)
    print(f"  Raw points      : {len(raw):,}")
    print(f"  After voxelize  : {len(vox):,}  points")

    # Semantic segmentation (Module 2.1)
    print(f"\n[Step 1] Semantic segmentation (RangeNet++) …")
    sem_labels = segment_point_cloud(vox)

    unique_sem, counts_sem = np.unique(sem_labels, return_counts=True)
    for cls, cnt in zip(unique_sem, counts_sem):
        print(f"         {CLASS_NAMES.get(cls, 'unknown'):<20} : {cnt:>6,} pts")

    # Instance segmentation (DBSCAN)
    print(f"\n[Step 2] DBSCAN instance segmentation …")
    print(f"         Parameters:")
    for cls_id, p in CLASS_DBSCAN_PARAMS.items():
        print(f"           {CLASS_NAMES[cls_id]:<14}  ε={p.eps} m  min_pts={p.min_samples}")

    inst_ids = instance_segment(vox, sem_labels)

    # Results
    noise_count = int(np.sum(inst_ids == NOISE_ID))
    n_instances = int(np.sum(inst_ids >= 0))
    n_clusters  = int(inst_ids.max()) + 1 if n_instances > 0 else 0

    print(f"\n{'─' * 60}")
    print(f"  Total instances detected : {n_clusters}")
    print(f"  Points assigned          : {n_instances:,}")
    print(f"  Noise / skipped          : {noise_count:,}")
    print(f"{'─' * 60}")

    rows = cluster_summary(sem_labels, inst_ids)

    # Group by class for a cleaner printout
    from itertools import groupby
    rows_sorted = sorted(rows, key=lambda r: (r["semantic_class"], -r["n_points"]))

    print(f"\n  {'Inst ID':>7}  {'Class':<18}  {'Points':>7}")
    print(f"  {'─' * 38}")
    for cls_id, group in groupby(rows_sorted, key=lambda r: r["semantic_class"]):
        cls_rows = list(group)
        print(f"\n  ── {CLASS_NAMES.get(cls_id, 'unknown')}  ({len(cls_rows)} instance(s)) ──")
        for r in cls_rows:
            bar = "█" * min(int(r["n_points"] / 30), 20)
            print(f"  {r['instance_id']:>7}  {r['class_name']:<18}  {r['n_points']:>7,}  {bar}")

    print()

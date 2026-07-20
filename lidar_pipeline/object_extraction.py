"""
object_extraction.py

Extract the points of every detected object instance and save each one
as an individual .npy file.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import load_bin_file, get_all_frames, default_velodyne_dir
from lidar_pipeline.voxelization import voxelize
from lidar_pipeline.semantic_segmentation import segment_point_cloud, CLASS_NAMES
from lidar_pipeline.instance_segmentation import instance_segment, cluster_summary, NOISE_ID


# Class-name → file-name slug  (matches the example: car_1.npy, pedestrian_2.npy)
CLASS_SLUG = {
    0: "unlabelled",
    1: "ground",
    2: "vegetation",
    3: "building",
    4: "car",
    5: "pedestrian",
}


def extract_objects(
    points:          np.ndarray,
    semantic_labels: np.ndarray,
    instance_ids:    np.ndarray,
    output_dir:      str | Path,
    frame_name:      str = "frame",
    min_points:      int = 3,
    rows:            list | None = None,
) -> list[Path]:

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) array, got shape {points.shape}")

    out_dir = Path(output_dir) / frame_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    if rows is None:
        rows = cluster_summary(semantic_labels, instance_ids)

    for obj in rows:
        if obj["n_points"] < min_points:
            continue

        inst_id = obj["instance_id"]
        cls_id  = obj["semantic_class"]
        slug    = CLASS_SLUG.get(cls_id, "object")
        filepath = out_dir / f"{slug}_{inst_id}.npy"

        mask = instance_ids == inst_id
        np.save(filepath, points[mask])
        saved.append(filepath)

    return saved


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    DATA_DIR = default_velodyne_dir()
    OUTPUT_DIR = Path("data/objects")

    print("\t Object Extraction")
    print("-" * 60)

    try:
        frames = get_all_frames(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    fp = frames[0]
    print(f"\nFrame : {fp.name}")

    raw = load_bin_file(fp)
    vox = voxelize(raw, voxel_size=0.1)
    print(f"  Raw points     : {len(raw):,}")
    print(f"  After voxelize : {len(vox):,}  points")

    print(f"\n[Step 1] Semantic segmentation …")
    sem_labels = segment_point_cloud(vox)

    print(f"[Step 2] Instance segmentation (DBSCAN) …")
    inst_ids = instance_segment(vox, sem_labels)

    n_instances = int((inst_ids >= 0).sum())
    n_clusters  = int(inst_ids.max()) + 1 if n_instances > 0 else 0
    print(f"         {n_clusters} instances found")

    print(f"\n[Step 3] Extracting and saving objects → {OUTPUT_DIR}/")
    frame_name  = fp.stem               # e.g. "000000"
    saved_paths = extract_objects(
        vox, sem_labels, inst_ids,
        output_dir=OUTPUT_DIR,
        frame_name=frame_name,
    )

    # Summary
    print(f"\n{'─' * 60}")
    print(f"  Files saved : {len(saved_paths)}")
    print(f"  Location    : {OUTPUT_DIR / frame_name}")
    print(f"{'─' * 60}")

    # Show first 10 files as examples
    examples = saved_paths[:10]
    print(f"\n  Example files:")
    for p in examples:
        arr = np.load(p)
        print(f"    {p.name:<35}  shape {arr.shape}")

    if len(saved_paths) > 10:
        print(f"    ... and {len(saved_paths) - 10} more files")

    print()

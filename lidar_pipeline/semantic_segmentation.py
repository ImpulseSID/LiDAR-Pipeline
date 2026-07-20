"""
semantic_segmentation.py

Assign a semantic class label to every point in a LiDAR point cloud
using the RangeNet++ algorithm (Milioto et al., IROS 2019).
"""

import sys
import math
import numpy as np
from pathlib import Path
from collections import Counter

# Ensure the project root is on sys.path so `lidar_pipeline` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import load_bin_file, get_all_frames, default_velodyne_dir
from lidar_pipeline.voxelization import voxelize
from lidar_pipeline.normalization import normalize


# Constants

# KITTI Velodyne parameters
FOV_UP_DEG   =  2.0
FOV_DOWN_DEG = -24.8
RANGE_IMG_H  =  64
RANGE_IMG_W  =  1024

# Semantic class IDs
CLASS_UNLABELLED  = 0
CLASS_GROUND      = 1
CLASS_VEGETATION  = 2
CLASS_BUILDING    = 3
CLASS_VEHICLE     = 4
CLASS_PERSON      = 5

CLASS_NAMES = {
    CLASS_UNLABELLED : "unlabelled",
    CLASS_GROUND     : "ground",
    CLASS_VEGETATION : "vegetation",
    CLASS_BUILDING   : "building",
    CLASS_VEHICLE    : "vehicle",
    CLASS_PERSON     : "person / cyclist",
}

# kNN back-projection
KNN_K       = 5
KNN_SEARCH  = 7


# Spherical Projection  (3-D → 2-D)

def spherical_projection(
    points: np.ndarray,
    H: int   = RANGE_IMG_H,
    W: int   = RANGE_IMG_W,
    fov_up_deg:   float = FOV_UP_DEG,
    fov_down_deg: float = FOV_DOWN_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Project a (N, 4) point cloud onto a spherical range image.

    
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) array, got shape {points.shape}")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Range
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)          # (N,)

    valid_mask = r > 1e-6

    # Spherical angles
    yaw   = np.arctan2(y, x)
    pitch = np.where(valid_mask, np.arcsin(np.clip(z / np.where(r > 0, r, 1), -1.0, 1.0)), 0.0)

    fov_up_rad   = math.radians(fov_up_deg)
    fov_down_rad = math.radians(fov_down_deg)
    fov_rad      = fov_up_rad - fov_down_rad

    # Image coordinates
    u_f = 0.5 * (1.0 - yaw / np.pi) * W
    v_f = (1.0 - (pitch - fov_down_rad) / fov_rad) * H

    # Clip & cast to integer pixel indices
    u = np.clip(np.floor(u_f).astype(np.int32), 0, W - 1)
    v = np.clip(np.floor(v_f).astype(np.int32), 0, H - 1)

    # Build range image (closest-point-wins)
    range_image = np.zeros((H, W, 5), dtype=np.float32)

    order          = np.argsort(-r)
    u_s, v_s       = u[order], v[order]
    r_s            = r[order]
    pts_s          = points[order]

    range_image[v_s, u_s, 0] = pts_s[:, 0]
    range_image[v_s, u_s, 1] = pts_s[:, 1]
    range_image[v_s, u_s, 2] = pts_s[:, 2]
    range_image[v_s, u_s, 3] = r_s
    range_image[v_s, u_s, 4] = pts_s[:, 3]

    return range_image, v, u


# Range-Image Segmentation

def _classify_pixel(
    z:         float,
    r:         float,
    intensity: float,
    row:       int,
    H:         int,
    fov_up_deg:   float,
    fov_down_deg: float,
) -> int:
    if r < 1e-3:
        return CLASS_UNLABELLED

    # Fractional position within the vertical FOV (0 = top beam, 1 = bottom)
    row_frac = row / max(H - 1, 1)

    # Ground / road
    # Bottom 30 % of scan lines, very near to ground plane (z < 0.4 m above sensor)
    if row_frac > 0.70 and z < 0.4 and r < 60.0:
        return CLASS_GROUND

    # Flat, far ground (long-range road surface)
    if row_frac > 0.55 and -1.8 < z < 0.2 and r > 5.0:
        return CLASS_GROUND

    # Building / large structure    
    if z > 1.5 and r < 50.0 and intensity < 0.25:
        return CLASS_BUILDING


    if -0.5 < z < 2.5 and 2.0 < r < 40.0 and intensity > 0.20:
        return CLASS_VEHICLE

    # Person / cyclist
    if -0.5 < z < 2.0 and r < 15.0 and intensity < 0.20:
        return CLASS_PERSON

    # Vegetation
    if z < 1.5 and intensity < 0.15:
        return CLASS_VEGETATION

    # Anything else → unlabelled
    return CLASS_UNLABELLED


def segment_range_image(
    range_image:  np.ndarray,
    fov_up_deg:   float = FOV_UP_DEG,
    fov_down_deg: float = FOV_DOWN_DEG,
) -> np.ndarray:
    H, W, _ = range_image.shape
    label_image = np.zeros((H, W), dtype=np.int32)

    z_img    = range_image[:, :, 2]
    r_img    = range_image[:, :, 3]
    int_img  = range_image[:, :, 4]

    valid = r_img > 1e-3

    # Vectorised extraction of features for valid pixels
    rows_v, cols_v = np.where(valid)
    z_v    = z_img  [rows_v, cols_v]
    r_v    = r_img  [rows_v, cols_v]
    int_v  = int_img[rows_v, cols_v]

    labels_v = np.array([
        _classify_pixel(
            float(z_v[i]),
            float(r_v[i]),
            float(int_v[i]),
            int(rows_v[i]),
            H,
            fov_up_deg,
            fov_down_deg,
        )
        for i in range(len(rows_v))
    ], dtype=np.int32)

    label_image[rows_v, cols_v] = labels_v

    return label_image

# kNN Label Back-Projection  (2-D labels → original 3-D points)

def knn_label_backproject(
    label_image: np.ndarray,
    pixel_row:   np.ndarray,
    pixel_col:   np.ndarray,
    k:           int = KNN_K,
    search_r:    int = KNN_SEARCH,
) -> np.ndarray:
    H, W        = label_image.shape
    N           = len(pixel_row)
    point_labels = np.zeros(N, dtype=np.int32)

    for i in range(N):
        v0, u0 = int(pixel_row[i]), int(pixel_col[i])

        # Search window (clamped to image bounds)
        v_lo = max(0,     v0 - search_r)
        v_hi = min(H - 1, v0 + search_r)
        u_lo = max(0,     u0 - search_r)
        u_hi = min(W - 1, u0 + search_r)

        patch = label_image[v_lo: v_hi + 1, u_lo: u_hi + 1]   # small window

        # Gather valid (non-unlabelled) labels
        valid_labels = patch[patch > CLASS_UNLABELLED].ravel()

        if len(valid_labels) == 0:
            # No labelled neighbour found — fall back to the pixel's own label
            point_labels[i] = int(label_image[v0, u0])
            continue

        top_k = valid_labels[:k] if len(valid_labels) >= k else valid_labels
        majority = Counter(top_k.tolist()).most_common(1)[0][0]
        point_labels[i] = majority

    return point_labels


# single-frame segmentation

def segment_point_cloud(
    points: np.ndarray,
    H: int   = RANGE_IMG_H,
    W: int   = RANGE_IMG_W,
    fov_up_deg:   float = FOV_UP_DEG,
    fov_down_deg: float = FOV_DOWN_DEG,
    knn_k:        int   = KNN_K,
    knn_search_r: int   = KNN_SEARCH,
) -> np.ndarray:
    
    # Spherical projection
    range_image, pixel_row, pixel_col = spherical_projection(
        points, H, W, fov_up_deg, fov_down_deg
    )

    # Range-image classification
    label_image = segment_range_image(range_image, fov_up_deg, fov_down_deg)

    # kNN back-projection
    labels = knn_label_backproject(
        label_image, pixel_row, pixel_col, k=knn_k, search_r=knn_search_r
    )

    return labels


# class-distribution summary

def class_distribution(labels: np.ndarray) -> dict:

    N = len(labels)
    dist = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        count = int(np.sum(labels == cls_id))
        dist[cls_name] = {
            "class_id": cls_id,
            "count":    count,
            "pct":      round(100.0 * count / N, 2) if N > 0 else 0.0,
        }
    return dist


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    DATA_DIR = default_velodyne_dir()

    print("\t RangeNet++ Semantic Segmentation")
    print("-" * 60)

    # Load
    try:
        frames = get_all_frames(DATA_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    fp = frames[0]
    print(f"\nFrame : {fp.name}")

    raw = load_bin_file(fp)
    print(f"  Raw points      : {len(raw):,}  |  shape {raw.shape}")

    # pre-processing
    vox  = voxelize(raw, voxel_size=0.1)
    norm = normalize(vox)
    print(f"  After voxelize  : {len(vox):,}  points")
    print(f"  After normalize : {len(norm):,}  points  (coords in [0, 1])")

    print(f"\n[Step 1] Spherical projection → ({RANGE_IMG_H} × {RANGE_IMG_W}) range image …")
    range_image, pixel_row, pixel_col = spherical_projection(vox)
    valid_pixels = int(np.sum(range_image[:, :, 3] > 1e-3))
    print(f"         Valid pixels : {valid_pixels:,}  / {RANGE_IMG_H * RANGE_IMG_W:,}")

    print(f"\n[Step 2] Segmenting range image …")
    label_image = segment_range_image(range_image)

    print(f"\n[Step 3] kNN back-projection (k={KNN_K}, search_r={KNN_SEARCH}) …")
    labels = knn_label_backproject(label_image, pixel_row, pixel_col)

    # Results
    print(f"\n{'─' * 60}")
    print(f"{'Class':<22} {'ID':>3}  {'Count':>8}  {'Pct':>7}")
    print(f"{'─' * 60}")
    dist = class_distribution(labels)
    for cls_name, info in dist.items():
        bar_len = int(info["pct"] / 2)
        bar = "█" * bar_len
        print(
            f"  {cls_name:<20} {info['class_id']:>3}  "
            f"{info['count']:>8,}  {info['pct']:>6.1f}%  {bar}"
        )
    print(f"{'─' * 60}")
    print(f"  {'TOTAL':<20} {'':>3}  {len(labels):>8,}  {'100.0':>6}%")
    print()
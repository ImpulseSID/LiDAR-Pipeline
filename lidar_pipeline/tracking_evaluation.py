"""
tracking_evaluation.py

Task 3 — Evaluate the impact of the POPA attack on 3-D multi-object tracking.

Pipeline (per KITTI *tracking* sequence):

    clean  .bin frames ─┐
                        ├─▶ PointPillars (OpenPCDet) ─▶ detections ─▶ 3-D Kalman
    adv    .bin frames ─┘                                             tracker
                                                                        │
    KITTI GT labels ────────────────────────────────────────────────┐  ▼
    KITTI calibration ──────────────────────────────────────────────┴▶ metrics
                                                                        │
                                          Table 1 / Table 2 + plots ◀───┘

The four metrics (each computed for the CLEAN and the POPA sequences and then
compared) are:

    1. Velocity Error       |v_estimated − v_groundtruth|   (mean / max)
    2. ID Switches          # of times a GT track changes tracker-id (total / avg)
    3. Trajectory Error     mean sqrt((x−x̂)² + (y−ŷ)²)      (mean / max, metres)
    4. Track Fragmentation  # of times a GT track is interrupted (total)

Everything lives in a single module and is driven by a CLI (see ``--help``).
Nothing is hard-coded to one sequence: pass ``--seq 0000`` or ``--seq all`` or
several ids, and point ``--adv-dir`` at either a flat folder of ``adv_XXXX.bin``
or a folder with per-sequence sub-folders (``<adv-dir>/<seq>/adv_XXXX.bin``).

Coordinate frames
-----------------
* KITTI GT boxes are stored in the *camera* frame as the box *bottom* centre.
  They are lifted to the box centre (``y_cam -= h/2``) and transformed into the
  *Velodyne / LiDAR* frame using the calibration (``R_rect``, ``Tr_velo_cam``).
* OpenPCDet's KITTI PointPillars predicts boxes already in the LiDAR frame as
  ``[x, y, z(centre), dx(=l), dy(=w), dz(=h), heading]``.
* Detections, GT and tracks therefore all live in the LiDAR frame, so IoU and
  association are consistent.

Velocity note
-------------
KITTI tracking here ships without ego-motion (oxts) poses, so both the GT
velocity (finite difference of GT centres) and the estimated velocity (Kalman
state) are expressed in the *moving sensor* frame. This is internally
consistent — relative velocities are compared to relative velocities — but is
not absolute world-frame velocity. Frame rate is 10 Hz, so ``dt = 0.1 s``.

Dependencies: numpy, scipy (optional, Hungarian), matplotlib, pandas, and
OpenPCDet + torch for the detector (installed on Colab; guarded so this module
imports fine without them).
"""

from __future__ import annotations

import os
import sys
import csv
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

# Ensure the project root is on sys.path so `lidar_pipeline` is importable when
# running:  python lidar_pipeline/tracking_evaluation.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import (
    PROJECT_ROOT,
    default_velodyne_dir,
    get_all_frames,
    load_bin_file,
    list_tracking_sequences,
    DEFAULT_SEQUENCE,
)

# Reuse the project's gate constant (kept consistent with attack_metric.py).
from lidar_pipeline.attack_metric import MATCH_GATE_M


# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

DT: float = 0.1  # KITTI @ 10 Hz

# KITTI tracking dataset layout (relative to the project root; all overridable).
DEFAULT_LABEL_DIR: Path = PROJECT_ROOT / "data" / "data_tracking_label_2" / "training" / "label_02"
DEFAULT_CALIB_DIR: Path = PROJECT_ROOT / "data" / "data_tracking_calib" / "training" / "calib"

# Object classes we evaluate (the KITTI PointPillars classes).
EVAL_CLASSES = ("Car", "Pedestrian", "Cyclist")
# KITTI GT type -> evaluation class (fold Van into Car; drop DontCare/Misc).
GT_TYPE_MAP = {
    "Car": "Car",
    "Van": "Car",
    "Truck": "Car",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
    "Cyclist": "Cyclist",
}

# Detection CSV schema (stable, shared by detector + tracker inputs).
DET_CSV_FIELDS = [
    "frame_id", "det_id", "class",
    "x", "y", "z", "l", "w", "h", "yaw",
    "score", "cx", "cy", "cz",
]

# Tracking-results CSV schema.
TRK_CSV_FIELDS = [
    "frame_id", "track_id", "x", "y", "z", "vx", "vy", "vz", "speed", "class",
]

# Tracker defaults (AB3DMOT-style).
DEFAULT_MIN_HITS = 3
DEFAULT_MAX_AGE = 2
DEFAULT_IOU_THRESH = 0.01     # BEV IoU minimum to accept a match
DEFAULT_GATE_M = MATCH_GATE_M  # max centre distance (m) for a match

# Detector default score threshold.
DEFAULT_SCORE_THRESH = 0.3


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — Ground-truth + calibration I/O and clean/adv frame resolver
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GtObject:
    """A ground-truth object in the LiDAR frame."""
    frame_id: int
    track_id: int
    cls: str
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def dims(self) -> np.ndarray:
        return np.array([self.l, self.w, self.h], dtype=np.float64)


def _read_calib(seq: str, calib_dir: Path) -> dict[str, np.ndarray]:
    """Parse a KITTI tracking calibration file into named matrices."""
    path = Path(calib_dir) / f"{seq}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    mats: dict[str, np.ndarray] = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            key, _, values = line.partition(":")
            key = key.strip()
            try:
                nums = np.array([float(v) for v in values.split()], dtype=np.float64)
            except ValueError:
                continue
            mats[key] = nums

    def _mat(key: str, rows: int, cols: int) -> np.ndarray:
        if key not in mats:
            raise KeyError(f"Calib key '{key}' missing in {path}")
        return mats[key].reshape(rows, cols)

    R_rect = _mat("R_rect", 3, 3)
    Tr_velo_cam = _mat("Tr_velo_cam", 3, 4)

    # 4x4 homogeneous forms.
    R_rect_h = np.eye(4, dtype=np.float64)
    R_rect_h[:3, :3] = R_rect

    Tr_h = np.eye(4, dtype=np.float64)
    Tr_h[:3, :4] = Tr_velo_cam

    return {
        "R_rect": R_rect,
        "Tr_velo_cam": Tr_velo_cam,
        "R_rect_h": R_rect_h,
        "Tr_velo_cam_h": Tr_h,
        # cam(rect) -> velo :  X_velo = Tr^{-1} @ R_rect^{-1} @ X_cam
        "cam_to_velo": np.linalg.inv(Tr_h) @ np.linalg.inv(R_rect_h),
    }


def _cam_to_velo(points_cam: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray:
    """Transform (N,3) points from the rectified camera frame to the LiDAR frame."""
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    hom = np.hstack([pts, np.ones((len(pts), 1))])          # (N,4)
    velo = (calib["cam_to_velo"] @ hom.T).T                 # (N,4)
    return velo[:, :3]


def load_gt_by_frame(seq: str, label_dir: Path, calib_dir: Path) -> dict[int, list[GtObject]]:
    """
    Parse ``label_02/<seq>.txt`` grouped by frame index, returning GT objects in
    LiDAR coordinates. Rows of type ``DontCare`` (and other non-eval types) are
    skipped. The box bottom-centre is lifted to the box centre before transform.
    """
    path = Path(label_dir) / f"{seq}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    calib = _read_calib(seq, calib_dir)
    by_frame: dict[int, list[GtObject]] = {}

    with open(path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 17:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            gt_type = parts[2]
            if gt_type not in GT_TYPE_MAP:
                continue  # DontCare, Misc, ...
            cls = GT_TYPE_MAP[gt_type]

            h, w, l = float(parts[10]), float(parts[11]), float(parts[12])
            x_cam, y_cam, z_cam = float(parts[13]), float(parts[14]), float(parts[15])
            ry = float(parts[16])

            # Lift bottom-centre -> 3-D box centre (camera Y points down).
            center_cam = np.array([[x_cam, y_cam - h / 2.0, z_cam]], dtype=np.float64)
            center_velo = _cam_to_velo(center_cam, calib)[0]

            # KITTI yaw (camera ry) -> LiDAR heading.
            yaw = -ry - math.pi / 2.0

            obj = GtObject(
                frame_id=frame_id, track_id=track_id, cls=cls,
                x=float(center_velo[0]), y=float(center_velo[1]), z=float(center_velo[2]),
                l=l, w=w, h=h, yaw=_normalize_angle(yaw),
            )
            by_frame.setdefault(frame_id, []).append(obj)

    return by_frame


@dataclass
class FramePair:
    """One tracking frame: its source index plus clean and adversarial paths."""
    frame_id: int
    clean_path: Path
    adv_path: Path
    is_attacked: bool  # False when the adv file is a byte-identical clean copy


def _adv_dir_for_seq(adv_dir: Path, seq: str) -> Path:
    """Return the per-sequence adv sub-folder if it exists, else the flat dir."""
    adv_dir = Path(adv_dir)
    per_seq = adv_dir / seq
    if per_seq.is_dir() and any(per_seq.glob("adv_*.bin")):
        return per_seq
    return adv_dir


def _adv_frame_index(adv_path: Path) -> Optional[int]:
    """Extract the source frame index from an ``adv_XXXX.bin`` filename."""
    stem = adv_path.stem  # adv_0007
    if not stem.startswith("adv_"):
        return None
    try:
        return int(stem[len("adv_"):])
    except ValueError:
        return None


def resolve_frames(
    seq: str,
    adv_dir: Path,
    velodyne_root: Optional[Path] = None,
) -> list[FramePair]:
    """
    Build the ordered list of :class:`FramePair` for a sequence by mapping each
    ``adv_XXXX.bin`` to its source clean frame ``NNNNNN.bin``.

    ``adv_dir`` may be flat (``<adv_dir>/adv_XXXX.bin``) or per-sequence
    (``<adv_dir>/<seq>/adv_XXXX.bin``); the per-sequence sub-folder wins when it
    exists.
    """
    clean_dir = Path(velodyne_root) / seq if velodyne_root else default_velodyne_dir(seq)
    seq_adv_dir = _adv_dir_for_seq(adv_dir, seq)

    adv_files = sorted(seq_adv_dir.glob("adv_*.bin"))
    if not adv_files:
        raise FileNotFoundError(f"No adv_*.bin files found in {seq_adv_dir}")

    pairs: list[FramePair] = []
    for adv_path in adv_files:
        idx = _adv_frame_index(adv_path)
        if idx is None:
            continue
        clean_path = clean_dir / f"{idx:06d}.bin"
        if not clean_path.exists():
            continue
        is_attacked = _bin_differs(clean_path, adv_path)
        pairs.append(FramePair(idx, clean_path, adv_path, is_attacked))

    pairs.sort(key=lambda p: p.frame_id)
    if not pairs:
        raise RuntimeError(
            f"Could not pair any adv frame in {seq_adv_dir} with a clean frame in {clean_dir}"
        )
    return pairs


def _bin_differs(a: Path, b: Path) -> bool:
    """True if two .bin files differ (cheap size check, then bytes)."""
    try:
        if a.stat().st_size != b.stat().st_size:
            return True
        return a.read_bytes() != b.read_bytes()
    except OSError:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Small geometry helpers (shared by tracker + metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _box_aabb(center: np.ndarray, l: float, w: float, h: float, yaw: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Axis-aligned bounding box of a rotated 3-D box, as (min_xyz, max_xyz).

    The horizontal footprint of a box of size (l, w) rotated by ``yaw`` has
    half-extents  ex = l/2|cos| + w/2|sin| ,  ey = l/2|sin| + w/2|cos|.
    """
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    ex = (l / 2.0) * c + (w / 2.0) * s
    ey = (l / 2.0) * s + (w / 2.0) * c
    ez = h / 2.0
    cen = np.asarray(center, dtype=np.float64)
    half = np.array([ex, ey, ez], dtype=np.float64)
    return cen - half, cen + half


def bev_iou(box_a: "np.ndarray | Sequence[float]", box_b: "np.ndarray | Sequence[float]") -> float:
    """
    Bird's-eye-view (x-y) IoU between two boxes given as
    ``[x, y, z, l, w, h, yaw]``, using their axis-aligned footprints (a robust
    approximation that, combined with a centre-distance gate, is sufficient for
    association).
    """
    ax, ay, az, al, aw, ah, ayaw = box_a
    bx, by, bz, bl, bw, bh, byaw = box_b
    amin, amax = _box_aabb(np.array([ax, ay, az]), al, aw, ah, ayaw)
    bmin, bmax = _box_aabb(np.array([bx, by, bz]), bl, bw, bh, byaw)

    lo = np.maximum(amin[:2], bmin[:2])
    hi = np.minimum(amax[:2], bmax[:2])
    inter_wh = np.maximum(hi - lo, 0.0)
    inter = float(inter_wh[0] * inter_wh[1])

    area_a = float((amax[0] - amin[0]) * (amax[1] - amin[1]))
    area_b = float((bmax[0] - bmin[0]) * (bmax[1] - bmin[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — PointPillars detector + serialized detections
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """A single 3-D detection in the LiDAR frame."""
    frame_id: int
    det_id: int
    cls: str
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float
    score: float
    cx: float
    cy: float
    cz: float

    @property
    def box7(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.l, self.w, self.h, self.yaw],
                        dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.cz], dtype=np.float64)

    def to_row(self) -> dict:
        return {
            "frame_id": self.frame_id, "det_id": self.det_id, "class": self.cls,
            "x": self.x, "y": self.y, "z": self.z,
            "l": self.l, "w": self.w, "h": self.h, "yaw": self.yaw,
            "score": self.score, "cx": self.cx, "cy": self.cy, "cz": self.cz,
        }

    @staticmethod
    def from_row(row: dict) -> "Detection":
        f = lambda k: float(row[k])
        return Detection(
            frame_id=int(float(row["frame_id"])), det_id=int(float(row["det_id"])),
            cls=str(row["class"]),
            x=f("x"), y=f("y"), z=f("z"), l=f("l"), w=f("w"), h=f("h"), yaw=f("yaw"),
            score=f("score"), cx=f("cx"), cy=f("cy"), cz=f("cz"),
        )


class Detector:
    """Minimal detector interface."""

    def detect(self, points: np.ndarray, frame_id: int = 0) -> list[Detection]:
        raise NotImplementedError


class PointPillarsDetector(Detector):
    """
    PointPillars detector backed by OpenPCDet's pretrained KITTI model.

    Heavy imports (``torch``, ``pcdet``) are performed lazily inside
    ``__init__`` so that merely importing this module (e.g. on a machine without
    a GPU) never fails. Instantiate this class only where OpenPCDet is
    installed (Colab).
    """

    def __init__(
        self,
        cfg_file: Optional[str] = None,
        ckpt: Optional[str] = None,
        score_thresh: float = DEFAULT_SCORE_THRESH,
        device: str = "cuda",
    ) -> None:
        cfg_file = cfg_file or os.environ.get("POINTPILLARS_CFG")
        ckpt = ckpt or os.environ.get("POINTPILLARS_CKPT")
        if not cfg_file or not ckpt:
            raise ValueError(
                "PointPillarsDetector needs a cfg file and checkpoint. Pass "
                "--cfg/--ckpt or set POINTPILLARS_CFG / POINTPILLARS_CKPT."
            )

        try:
            import torch  # noqa: F401
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.datasets import DatasetTemplate
            from pcdet.models import build_network, load_data_to_gpu
            from pcdet.utils import common_utils
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "OpenPCDet / torch are required for PointPillarsDetector. Install "
                "them on Colab (see summer_intern.ipynb setup cell)."
            ) from exc

        self._torch = torch
        self._load_data_to_gpu = load_data_to_gpu
        self.score_thresh = float(score_thresh)
        self.device = device

        logger = common_utils.create_logger()
        cfg_from_yaml_file(cfg_file, cfg)
        self.class_names = list(cfg.CLASS_NAMES)

        # A tiny inference-only dataset that carries the model's point-feature
        # encoding and data-processing config (mirrors OpenPCDet's demo.py).
        class _InferDataset(DatasetTemplate):
            def __init__(self, dataset_cfg, class_names):
                super().__init__(
                    dataset_cfg=dataset_cfg, class_names=class_names,
                    training=False, root_path=None, logger=logger,
                )

        self._dataset = _InferDataset(cfg.DATA_CONFIG, self.class_names)

        self.model = build_network(
            model_cfg=cfg.MODEL, num_class=len(self.class_names), dataset=self._dataset,
        )
        self.model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=(device != "cuda"))
        if device == "cuda":
            self.model.cuda()
        self.model.eval()

    def detect(self, points: np.ndarray, frame_id: int = 0) -> list[Detection]:
        torch = self._torch
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 4)

        data_dict = {"points": pts, "frame_id": frame_id}
        data_dict = self._dataset.prepare_data(data_dict=data_dict)
        data_dict = self._dataset.collate_batch([data_dict])
        self._load_data_to_gpu(data_dict)

        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()   # (N,7): x,y,z,dx,dy,dz,heading
        scores = pred_dicts[0]["pred_scores"].cpu().numpy()  # (N,)
        labels = pred_dicts[0]["pred_labels"].cpu().numpy()  # (N,) 1-indexed

        dets: list[Detection] = []
        det_id = 0
        for box, score, label in zip(boxes, scores, labels):
            if float(score) < self.score_thresh:
                continue
            cls = self.class_names[int(label) - 1] if 0 < int(label) <= len(self.class_names) else "Unknown"
            x, y, z, dx, dy, dz, heading = [float(v) for v in box]
            dets.append(Detection(
                frame_id=int(frame_id), det_id=det_id, cls=cls,
                x=x, y=y, z=z, l=dx, w=dy, h=dz, yaw=_normalize_angle(heading),
                score=float(score), cx=x, cy=y, cz=z,
            ))
            det_id += 1
        return dets


def write_detections_csv(path: Path, dets_by_frame: dict[int, list[Detection]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DET_CSV_FIELDS)
        writer.writeheader()
        for frame_id in sorted(dets_by_frame):
            for det in dets_by_frame[frame_id]:
                writer.writerow(det.to_row())


def read_detections_csv(path: Path) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    with open(path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            det = Detection.from_row(row)
            out.setdefault(det.frame_id, []).append(det)
    return out


def run_detector_over_frames(
    detector: Detector,
    frames: list[FramePair],
    condition: str,
    verbose: bool = True,
) -> dict[int, list[Detection]]:
    """Run a detector over the clean or adversarial version of every frame."""
    assert condition in ("clean", "adv")
    out: dict[int, list[Detection]] = {}
    for i, fp in enumerate(frames):
        path = fp.clean_path if condition == "clean" else fp.adv_path
        points = load_bin_file(path)
        dets = detector.detect(points, frame_id=fp.frame_id)
        out[fp.frame_id] = dets
        if verbose and (i % 20 == 0 or i == len(frames) - 1):
            print(f"    [{condition}] frame {fp.frame_id:>6}  "
                  f"({i + 1}/{len(frames)})  dets={len(dets)}", flush=True)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — AB3DMOT-style 3-D Kalman tracker + association + results CSVs
# ═════════════════════════════════════════════════════════════════════════════

class KalmanBox3D:
    """
    Constant-velocity 3-D Kalman filter for a single tracked box.

    State  (10): [x, y, z, l, w, h, yaw, vx, vy, vz]   (yaw at index 6)
    Measure (7): [x, y, z, l, w, h, yaw]   (== Detection.box7)
    """

    _count = 0

    def __init__(self, det: Detection, dt: float = DT):
        self.dt = dt

        # State transition: position += velocity * dt.
        F = np.eye(10, dtype=np.float64)
        F[0, 7] = dt
        F[1, 8] = dt
        F[2, 9] = dt
        self.F = F

        # Measurement matrix: observe the first 7 state components.
        H = np.zeros((7, 10), dtype=np.float64)
        H[:7, :7] = np.eye(7)
        self.H = H

        # Covariances (AB3DMOT-style: large initial uncertainty on velocity).
        self.P = np.eye(10, dtype=np.float64)
        self.P[7:, 7:] *= 1000.0
        self.P *= 10.0

        self.Q = np.eye(10, dtype=np.float64)
        self.Q[7:, 7:] *= 0.01

        self.R = np.eye(7, dtype=np.float64)

        z = det.box7
        self.x = np.zeros((10, 1), dtype=np.float64)
        self.x[:7, 0] = z

        # Bookkeeping.
        KalmanBox3D._count += 1
        self.id = KalmanBox3D._count
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.cls = det.cls
        self.last_score = det.score

    # -- filter -----------------------------------------------------------------
    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.x[6, 0] = _normalize_angle(self.x[6, 0])
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.x[:7, 0].copy()

    def update(self, det: Detection) -> None:
        z = det.box7.reshape(7, 1).copy()

        # Orientation correction: keep measured yaw within pi of the prediction.
        pred_yaw = self.x[6, 0]
        meas_yaw = z[6, 0]
        diff = _normalize_angle(meas_yaw - pred_yaw)
        z[6, 0] = pred_yaw + diff

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6, 0] = _normalize_angle(self.x[6, 0])
        self.P = (np.eye(10) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.cls = det.cls
        self.last_score = det.score

    # -- accessors --------------------------------------------------------------
    @property
    def box7(self) -> np.ndarray:
        return self.x[:7, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[7:10, 0].copy()


def _greedy_match(
    tracks_boxes: list[np.ndarray],
    det_boxes: list[np.ndarray],
    iou_thresh: float,
    gate_m: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Associate detections to tracks. Uses Hungarian (scipy) on an IoU cost when
    available, else greedy; both are gated by BEV IoU and centre distance.

    Returns (matches[(track_idx, det_idx)], unmatched_track_idx, unmatched_det_idx).
    """
    T, D = len(tracks_boxes), len(det_boxes)
    if T == 0 or D == 0:
        return [], list(range(T)), list(range(D))

    iou = np.zeros((T, D), dtype=np.float64)
    dist = np.zeros((T, D), dtype=np.float64)
    for ti, tb in enumerate(tracks_boxes):
        for di, db in enumerate(det_boxes):
            iou[ti, di] = bev_iou(tb, db)
            dist[ti, di] = float(np.linalg.norm(tb[:3] - db[:3]))

    valid = (iou >= iou_thresh) & (dist <= gate_m)

    matches: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment
        cost = 1.0 - iou
        cost[~valid] = 1e6
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if valid[r, c]:
                matches.append((int(r), int(c)))
    except Exception:
        # Greedy fallback: highest IoU first.
        order = np.argsort(-iou, axis=None)
        used_t, used_d = set(), set()
        for flat in order:
            ti, di = divmod(int(flat), D)
            if ti in used_t or di in used_d or not valid[ti, di]:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))

    matched_t = {m[0] for m in matches}
    matched_d = {m[1] for m in matches}
    unmatched_t = [i for i in range(T) if i not in matched_t]
    unmatched_d = [i for i in range(D) if i not in matched_d]
    return matches, unmatched_t, unmatched_d


def run_tracker(
    detections_by_frame: dict[int, list[Detection]],
    dt: float = DT,
    min_hits: int = DEFAULT_MIN_HITS,
    max_age: int = DEFAULT_MAX_AGE,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    gate_m: float = DEFAULT_GATE_M,
) -> dict[int, list[dict]]:
    """
    Run the AB3DMOT-style tracker over a sequence of per-frame detections.

    Returns ``{frame_id: [track_output, ...]}`` where each track_output is
    ``{track_id, center(np3), box(np7), velocity(np3), speed, class}``.
    """
    # Reset global track-id counter so clean/adv runs start from 1 independently.
    KalmanBox3D._count = 0

    tracks: list[KalmanBox3D] = []
    outputs: dict[int, list[dict]] = {}

    for frame_id in sorted(detections_by_frame):
        dets = detections_by_frame[frame_id]

        # 1. predict existing tracks.
        track_boxes = [t.predict() for t in tracks]
        det_boxes = [d.box7 for d in dets]

        # 2. associate.
        matches, unmatched_t, unmatched_d = _greedy_match(
            track_boxes, det_boxes, iou_thresh, gate_m,
        )

        # 3. update matched tracks.
        for ti, di in matches:
            tracks[ti].update(dets[di])

        # 4. birth new tracks from unmatched detections.
        for di in unmatched_d:
            tracks.append(KalmanBox3D(dets[di], dt=dt))

        # 5. cull dead tracks.
        tracks = [t for t in tracks if t.time_since_update <= max_age]

        # 6. emit confirmed tracks that were updated this frame.
        frame_out: list[dict] = []
        for t in tracks:
            confirmed = (t.hits >= min_hits) or (frame_id < min_hits)
            if t.time_since_update == 0 and confirmed:
                box = t.box7
                vel = t.velocity
                frame_out.append({
                    "track_id": t.id,
                    "center": box[:3].copy(),
                    "box": box,
                    "velocity": vel,
                    "speed": float(np.linalg.norm(vel)),
                    "class": t.cls,
                })
        outputs[frame_id] = frame_out

    return outputs


def write_tracking_csv(path: Path, tracks_by_frame: dict[int, list[dict]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRK_CSV_FIELDS)
        writer.writeheader()
        for frame_id in sorted(tracks_by_frame):
            for tr in tracks_by_frame[frame_id]:
                c = tr["center"]
                v = tr["velocity"]
                writer.writerow({
                    "frame_id": frame_id, "track_id": tr["track_id"],
                    "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
                    "vx": float(v[0]), "vy": float(v[1]), "vz": float(v[2]),
                    "speed": tr["speed"], "class": tr["class"],
                })


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4 — GT association + the four tracking metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_gt_velocity(gt_by_frame: dict[int, list[GtObject]], dt: float = DT
                        ) -> dict[int, dict[int, np.ndarray]]:
    """
    GT velocity per (track_id, frame) via finite difference of LiDAR-frame
    centres. Returns ``{track_id: {frame_id: velocity_vec(3,)}}``.

    Velocities are in the moving sensor frame (no ego-motion compensation).
    """
    # Gather per-track (frame -> center).
    by_track: dict[int, dict[int, np.ndarray]] = {}
    for frame_id, objs in gt_by_frame.items():
        for o in objs:
            by_track.setdefault(o.track_id, {})[frame_id] = o.center

    vel: dict[int, dict[int, np.ndarray]] = {}
    for track_id, centers in by_track.items():
        frames = sorted(centers)
        vel[track_id] = {}
        for i, f in enumerate(frames):
            if i > 0:
                prev = frames[i - 1]
                gap = f - prev
                v = (centers[f] - centers[prev]) / (dt * max(gap, 1))
            elif len(frames) > 1:
                nxt = frames[i + 1]
                gap = nxt - f
                v = (centers[nxt] - centers[f]) / (dt * max(gap, 1))
            else:
                v = np.zeros(3, dtype=np.float64)
            vel[track_id][f] = v
    return vel


def _associate_tracks_to_gt(
    gt_objs: list[GtObject],
    track_outs: list[dict],
    iou_thresh: float,
    gate_m: float,
) -> dict[int, int]:
    """
    Per-frame association GT->tracker. Returns ``{gt_track_id: tracker_id}`` for
    matched GT objects only.
    """
    if not gt_objs or not track_outs:
        return {}

    gt_boxes = [np.array([o.x, o.y, o.z, o.l, o.w, o.h, o.yaw]) for o in gt_objs]
    tr_boxes = [t["box"] for t in track_outs]

    matches, _, _ = _greedy_match(gt_boxes, tr_boxes, iou_thresh, gate_m)
    out: dict[int, int] = {}
    for gi, ti in matches:
        out[gt_objs[gi].track_id] = track_outs[ti]["track_id"]
    return out


@dataclass
class TrackTimeline:
    """Per-GT-track record used for metrics and plotting."""
    gt_track_id: int
    cls: str
    frames: list[int] = field(default_factory=list)          # frames GT is present
    matched_tracker_id: dict[int, Optional[int]] = field(default_factory=dict)
    gt_center: dict[int, np.ndarray] = field(default_factory=dict)
    trk_center: dict[int, Optional[np.ndarray]] = field(default_factory=dict)
    gt_speed: dict[int, float] = field(default_factory=dict)
    trk_speed: dict[int, Optional[float]] = field(default_factory=dict)


def build_timelines(
    gt_by_frame: dict[int, list[GtObject]],
    tracks_by_frame: dict[int, list[dict]],
    gt_vel: dict[int, dict[int, np.ndarray]],
    iou_thresh: float = DEFAULT_IOU_THRESH,
    gate_m: float = DEFAULT_GATE_M,
) -> dict[int, TrackTimeline]:
    """Assemble per-GT-track timelines of matches, positions and speeds."""
    timelines: dict[int, TrackTimeline] = {}

    for frame_id in sorted(gt_by_frame):
        gt_objs = gt_by_frame[frame_id]
        track_outs = tracks_by_frame.get(frame_id, [])
        assoc = _associate_tracks_to_gt(gt_objs, track_outs, iou_thresh, gate_m)
        trk_by_id = {t["track_id"]: t for t in track_outs}

        for o in gt_objs:
            tl = timelines.setdefault(o.track_id, TrackTimeline(o.track_id, o.cls))
            tl.frames.append(frame_id)
            tl.gt_center[frame_id] = o.center
            tl.gt_speed[frame_id] = float(np.linalg.norm(gt_vel.get(o.track_id, {}).get(frame_id, np.zeros(3))))

            matched_tid = assoc.get(o.track_id)
            tl.matched_tracker_id[frame_id] = matched_tid
            if matched_tid is not None:
                tr = trk_by_id[matched_tid]
                tl.trk_center[frame_id] = tr["center"]
                tl.trk_speed[frame_id] = tr["speed"]
            else:
                tl.trk_center[frame_id] = None
                tl.trk_speed[frame_id] = None

    return timelines


def compute_metrics(timelines: dict[int, TrackTimeline], n_sequences: int = 1) -> dict:
    """
    Compute the four tracking metrics from per-GT-track timelines.

    Returns a dict with mean/max velocity error, total/avg ID switches, mean/max
    trajectory error, and total fragmentation, plus the raw samples.
    """
    vel_errors: list[float] = []
    traj_errors: list[float] = []
    id_switches = 0
    fragmentations = 0

    for tl in timelines.values():
        frames = sorted(tl.frames)

        # Velocity error + trajectory error over matched frames.
        for f in frames:
            tc = tl.trk_center.get(f)
            if tc is None:
                continue
            gc = tl.gt_center[f]
            traj_errors.append(float(np.linalg.norm(gc[:2] - tc[:2])))
            ts = tl.trk_speed.get(f)
            if ts is not None:
                vel_errors.append(abs(ts - tl.gt_speed.get(f, 0.0)))

        # ID switches: consecutive matched frames whose tracker id changed.
        prev_tid = None
        for f in frames:
            tid = tl.matched_tracker_id.get(f)
            if tid is not None:
                if prev_tid is not None and tid != prev_tid:
                    id_switches += 1
                prev_tid = tid

        # Fragmentation: number of times coverage resumes after a gap
        # (matched -> unmatched -> matched) within the tracked span.
        status = [1 if tl.matched_tracker_id.get(f) is not None else 0 for f in frames]
        # trim leading/trailing unmatched (track not yet born / already gone).
        first = next((i for i, s in enumerate(status) if s == 1), None)
        last = next((len(status) - 1 - i for i, s in enumerate(reversed(status)) if s == 1), None)
        if first is not None and last is not None:
            core = status[first:last + 1]
            # count 0->1 transitions inside the span (each = one resumption).
            for i in range(1, len(core)):
                if core[i] == 1 and core[i - 1] == 0:
                    fragmentations += 1

    def _mean(xs): return float(np.mean(xs)) if xs else 0.0
    def _max(xs): return float(np.max(xs)) if xs else 0.0

    return {
        "velocity_error_mean": _mean(vel_errors),
        "velocity_error_max": _max(vel_errors),
        "id_switches_total": int(id_switches),
        "id_switches_avg_per_seq": float(id_switches) / max(n_sequences, 1),
        "trajectory_error_mean": _mean(traj_errors),
        "trajectory_error_max": _max(traj_errors),
        "fragmentation_total": int(fragmentations),
        "n_gt_tracks": len(timelines),
        "n_vel_samples": len(vel_errors),
        "n_traj_samples": len(traj_errors),
    }


# ═════════════════════════════════════════════════════════════════════════════
# TASK 5 — Reporting tables + visualization
# ═════════════════════════════════════════════════════════════════════════════

METRIC_ROWS = [
    ("Velocity Error (mean)", "velocity_error_mean", "m/s"),
    ("Velocity Error (max)", "velocity_error_max", "m/s"),
    ("ID Switches (total)", "id_switches_total", ""),
    ("ID Switches (avg/seq)", "id_switches_avg_per_seq", ""),
    ("Trajectory Error (mean)", "trajectory_error_mean", "m"),
    ("Trajectory Error (max)", "trajectory_error_max", "m"),
    ("Track Fragmentation (total)", "fragmentation_total", ""),
]


def build_table1(clean: dict, adv: dict) -> "object":
    """Aggregate comparison table (clean vs POPA). Returns a pandas DataFrame."""
    import pandas as pd
    rows = []
    for label, key, unit in METRIC_ROWS:
        rows.append({
            "Metric": label,
            "Unit": unit,
            "Clean": round(clean.get(key, 0.0), 4),
            "POPA": round(adv.get(key, 0.0), 4),
        })
    return pd.DataFrame(rows, columns=["Metric", "Unit", "Clean", "POPA"])


def build_table2(per_seq: dict[str, dict[str, dict]]) -> "object":
    """
    Sequence-wise table. ``per_seq`` maps seq -> {"clean": metrics, "adv": metrics}.
    Returns a pandas DataFrame.
    """
    import pandas as pd
    rows = []
    for seq in sorted(per_seq):
        c = per_seq[seq]["clean"]
        a = per_seq[seq]["adv"]
        rows.append({
            "Sequence": seq,
            "VelErr Clean": round(c["velocity_error_mean"], 3),
            "VelErr POPA": round(a["velocity_error_mean"], 3),
            "IDsw Clean": c["id_switches_total"],
            "IDsw POPA": a["id_switches_total"],
            "TrajErr Clean": round(c["trajectory_error_mean"], 3),
            "TrajErr POPA": round(a["trajectory_error_mean"], 3),
            "Frag Clean": c["fragmentation_total"],
            "Frag POPA": a["fragmentation_total"],
        })
    return pd.DataFrame(rows)


def save_table(df: "object", out_dir: Path, name: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    with open(out_dir / f"{name}.md", "w", encoding="utf-8") as fh:
        fh.write(md + "\n")


def _pick_longest_tracks(timelines: dict[int, TrackTimeline], n: int) -> list[int]:
    order = sorted(timelines, key=lambda k: len(timelines[k].frames), reverse=True)
    return order[:n]


def plot_all(
    clean_timelines: dict[int, TrackTimeline],
    adv_timelines: dict[int, TrackTimeline],
    out_dir: Path,
    seq: str,
) -> list[Path]:
    """Render the four PDF-required plot types as PNGs. Returns saved paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # (1) Clean vs POPA BEV trajectory plots for a few long GT tracks.
    track_ids = _pick_longest_tracks(clean_timelines, 4)
    if track_ids:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        for ax, tls, title in ((axes[0], clean_timelines, "Clean"),
                               (axes[1], adv_timelines, "POPA")):
            for tid in track_ids:
                tl = tls.get(tid)
                if tl is None:
                    continue
                frames = sorted(tl.frames)
                gx = [tl.gt_center[f][0] for f in frames]
                gy = [tl.gt_center[f][1] for f in frames]
                ax.plot(gx, gy, "--", alpha=0.5, label=f"GT {tid}")
                tf = [f for f in frames if tl.trk_center.get(f) is not None]
                tx = [tl.trk_center[f][0] for f in tf]
                ty = [tl.trk_center[f][1] for f in tf]
                ax.plot(tx, ty, "-o", ms=2, label=f"trk GT{tid}")
            ax.set_title(f"{title} — BEV trajectories (seq {seq})")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"seq{seq}_trajectories.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    # (2) Velocity-over-time for the longest GT track.
    if track_ids:
        tid = track_ids[0]
        fig, ax = plt.subplots(figsize=(11, 5))
        for tls, label, style in ((clean_timelines, "Clean est", "-"),
                                  (adv_timelines, "POPA est", "-")):
            tl = tls.get(tid)
            if tl is None:
                continue
            frames = sorted(tl.frames)
            fv = [f for f in frames if tl.trk_speed.get(f) is not None]
            sv = [tl.trk_speed[f] for f in fv]
            ax.plot(fv, sv, style, label=f"{label} (trk id changes shown)")
        # GT speed from clean timeline (same GT).
        tl = clean_timelines.get(tid)
        if tl is not None:
            frames = sorted(tl.frames)
            ax.plot(frames, [tl.gt_speed[f] for f in frames], "k--", label="GT speed")
        ax.set_title(f"Velocity over time — GT track {tid} (seq {seq})")
        ax.set_xlabel("frame")
        ax.set_ylabel("speed (m/s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"seq{seq}_velocity_over_time.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    # (3) ID-switch examples: assigned tracker id vs frame (clean vs POPA).
    def _switch_count(tl: TrackTimeline) -> int:
        prev, n = None, 0
        for f in sorted(tl.frames):
            tid_ = tl.matched_tracker_id.get(f)
            if tid_ is not None:
                if prev is not None and tid_ != prev:
                    n += 1
                prev = tid_
        return n

    adv_switch_tid = max(adv_timelines, key=lambda k: _switch_count(adv_timelines[k]), default=None)
    if adv_switch_tid is not None:
        fig, ax = plt.subplots(figsize=(11, 5))
        for tls, label, marker in ((clean_timelines, "Clean", "o"),
                                   (adv_timelines, "POPA", "x")):
            tl = tls.get(adv_switch_tid)
            if tl is None:
                continue
            frames = sorted(tl.frames)
            fv = [f for f in frames if tl.matched_tracker_id.get(f) is not None]
            ids = [tl.matched_tracker_id[f] for f in fv]
            ax.scatter(fv, ids, marker=marker, label=f"{label} (switches={_switch_count(tl)})")
        ax.set_title(f"Tracker id assigned to GT track {adv_switch_tid} (seq {seq})")
        ax.set_xlabel("frame")
        ax.set_ylabel("assigned tracker id")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"seq{seq}_id_switches.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    # (4) Fragmentation examples: presence timeline (matched 1/0) clean vs POPA.
    def _frag_count(tl: TrackTimeline) -> int:
        frames = sorted(tl.frames)
        status = [1 if tl.matched_tracker_id.get(f) is not None else 0 for f in frames]
        first = next((i for i, s in enumerate(status) if s == 1), None)
        last = next((len(status) - 1 - i for i, s in enumerate(reversed(status)) if s == 1), None)
        n = 0
        if first is not None and last is not None:
            core = status[first:last + 1]
            for i in range(1, len(core)):
                if core[i] == 1 and core[i - 1] == 0:
                    n += 1
        return n

    adv_frag_tid = max(adv_timelines, key=lambda k: _frag_count(adv_timelines[k]), default=None)
    if adv_frag_tid is not None:
        fig, ax = plt.subplots(figsize=(11, 4))
        for offset, (tls, label) in enumerate(((clean_timelines, "Clean"),
                                               (adv_timelines, "POPA"))):
            tl = tls.get(adv_frag_tid)
            if tl is None:
                continue
            frames = sorted(tl.frames)
            status = [(1 if tl.matched_tracker_id.get(f) is not None else 0) + offset * 1.5
                      for f in frames]
            ax.step(frames, status, where="mid", label=f"{label} (frag={_frag_count(tl)})")
        ax.set_title(f"Track presence timeline — GT track {adv_frag_tid} (seq {seq})")
        ax.set_xlabel("frame")
        ax.set_ylabel("matched (offset per condition)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"seq{seq}_fragmentation.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    return saved


# ═════════════════════════════════════════════════════════════════════════════
# TASK 6 — CLI orchestration
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_seq_list(seq_arg: Sequence[str]) -> list[str]:
    if len(seq_arg) == 1 and seq_arg[0].lower() == "all":
        return list_tracking_sequences()
    return list(seq_arg)


def evaluate_sequence(
    seq: str,
    adv_dir: Path,
    out_dir: Path,
    detector: Detector,
    label_dir: Path,
    calib_dir: Path,
    velodyne_root: Optional[Path],
    tracker_kwargs: dict,
    make_plots: bool = True,
) -> dict:
    """Full per-sequence pipeline. Returns {'clean': metrics, 'adv': metrics}."""
    print(f"\n=== Sequence {seq} ===", flush=True)
    seq_out = Path(out_dir) / seq
    seq_out.mkdir(parents=True, exist_ok=True)

    frames = resolve_frames(seq, adv_dir, velodyne_root)
    n_attacked = sum(1 for f in frames if f.is_attacked)
    print(f"  frames: {len(frames)}  (attacked={n_attacked}, clean-copies={len(frames) - n_attacked})",
          flush=True)

    # Ground truth + GT velocity.
    gt_by_frame = load_gt_by_frame(seq, label_dir, calib_dir)
    gt_vel = compute_gt_velocity(gt_by_frame, dt=DT)

    results: dict[str, dict] = {}
    timelines_by_cond: dict[str, dict[int, TrackTimeline]] = {}

    for condition in ("clean", "adv"):
        print(f"  detecting [{condition}] ...", flush=True)
        dets = run_detector_over_frames(detector, frames, condition)
        write_detections_csv(seq_out / f"{condition}_detections.csv", dets)

        print(f"  tracking  [{condition}] ...", flush=True)
        tracks = run_tracker(dets, dt=DT, **tracker_kwargs)
        write_tracking_csv(seq_out / f"{condition}_tracking_results.csv", tracks)

        timelines = build_timelines(
            gt_by_frame, tracks, gt_vel,
            iou_thresh=tracker_kwargs["iou_thresh"], gate_m=tracker_kwargs["gate_m"],
        )
        timelines_by_cond[condition] = timelines
        results[condition] = compute_metrics(timelines, n_sequences=1)

    if make_plots:
        saved = plot_all(timelines_by_cond["clean"], timelines_by_cond["adv"], seq_out, seq)
        print(f"  plots: {', '.join(p.name for p in saved)}", flush=True)

    return results


def _aggregate(per_seq: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Aggregate per-sequence metrics into an overall clean/adv summary."""
    agg = {"clean": {}, "adv": {}}
    n = len(per_seq)
    for cond in ("clean", "adv"):
        vel_means, vel_maxes, traj_means, traj_maxes = [], [], [], []
        idsw, frag = 0, 0
        for seq in per_seq:
            m = per_seq[seq][cond]
            vel_means.append(m["velocity_error_mean"])
            vel_maxes.append(m["velocity_error_max"])
            traj_means.append(m["trajectory_error_mean"])
            traj_maxes.append(m["trajectory_error_max"])
            idsw += m["id_switches_total"]
            frag += m["fragmentation_total"]
        agg[cond] = {
            "velocity_error_mean": float(np.mean(vel_means)) if vel_means else 0.0,
            "velocity_error_max": float(np.max(vel_maxes)) if vel_maxes else 0.0,
            "id_switches_total": int(idsw),
            "id_switches_avg_per_seq": float(idsw) / max(n, 1),
            "trajectory_error_mean": float(np.mean(traj_means)) if traj_means else 0.0,
            "trajectory_error_max": float(np.max(traj_maxes)) if traj_maxes else 0.0,
            "fragmentation_total": int(frag),
        }
    return agg


def cmd_inspect(args: argparse.Namespace) -> None:
    """Task-1 demo: print GT tracks (LiDAR coords) and the adv/clean mapping."""
    seqs = _resolve_seq_list(args.seq)
    for seq in seqs:
        print(f"\n=== Sequence {seq} — GT / frame inspection ===")
        frames = resolve_frames(seq, Path(args.adv_dir),
                                Path(args.velodyne_root) if args.velodyne_root else None)
        print(f"Frames paired: {len(frames)}  "
              f"(attacked={sum(f.is_attacked for f in frames)})")
        print("First 8 frame mappings (frame_id -> clean | adv | attacked):")
        for fp in frames[:8]:
            print(f"  {fp.frame_id:>6}  {fp.clean_path.name}  {fp.adv_path.name}  "
                  f"attacked={fp.is_attacked}")

        gt = load_gt_by_frame(seq, Path(args.label_dir), Path(args.calib_dir))
        show_frame = frames[0].frame_id
        print(f"\nGT objects at frame {show_frame} (LiDAR coords):")
        for o in gt.get(show_frame, []):
            print(f"  track {o.track_id:>3}  {o.cls:<10}  "
                  f"c=({o.x:6.2f},{o.y:6.2f},{o.z:6.2f})  "
                  f"lwh=({o.l:.2f},{o.w:.2f},{o.h:.2f})  yaw={o.yaw:+.2f}")


def cmd_run(args: argparse.Namespace) -> None:
    seqs = _resolve_seq_list(args.seq)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = PointPillarsDetector(
        cfg_file=args.cfg, ckpt=args.ckpt,
        score_thresh=args.score_thresh, device=args.device,
    )

    tracker_kwargs = {
        "min_hits": args.min_hits,
        "max_age": args.max_age,
        "iou_thresh": args.iou_thresh,
        "gate_m": args.gate,
    }

    per_seq: dict[str, dict[str, dict]] = {}
    for seq in seqs:
        per_seq[seq] = evaluate_sequence(
            seq=seq,
            adv_dir=Path(args.adv_dir),
            out_dir=out_dir,
            detector=detector,
            label_dir=Path(args.label_dir),
            calib_dir=Path(args.calib_dir),
            velodyne_root=Path(args.velodyne_root) if args.velodyne_root else None,
            tracker_kwargs=tracker_kwargs,
            make_plots=not args.no_plots,
        )

    # Aggregate + tables.
    agg = _aggregate(per_seq)
    table1 = build_table1(agg["clean"], agg["adv"])
    table2 = build_table2(per_seq)

    save_table(table1, out_dir, "table1_aggregate")
    save_table(table2, out_dir, "table2_sequence_wise")

    print("\n" + "=" * 70)
    print("TABLE 1 — Tracking performance (Clean vs POPA, aggregate)")
    print("=" * 70)
    print(table1.to_string(index=False))
    print("\n" + "=" * 70)
    print("TABLE 2 — Sequence-wise results")
    print("=" * 70)
    print(table2.to_string(index=False))
    print(f"\nArtifacts written under: {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracking_evaluation",
        description="Evaluate POPA attack impact on 3-D object tracking "
                    "(PointPillars + AB3DMOT-style Kalman tracker).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Shared path options.
    def _add_paths(sp):
        sp.add_argument("--seq", nargs="+", default=[DEFAULT_SEQUENCE],
                        help="Sequence id(s), e.g. 0000 0001, or 'all'.")
        sp.add_argument("--adv-dir", default="data/adversarial",
                        help="Adversarial frames dir (flat or per-seq subfolders).")
        sp.add_argument("--label-dir", default=str(DEFAULT_LABEL_DIR),
                        help="KITTI tracking label_02 dir.")
        sp.add_argument("--calib-dir", default=str(DEFAULT_CALIB_DIR),
                        help="KITTI tracking calib dir.")
        sp.add_argument("--velodyne-root", default=None,
                        help="Override root holding <seq>/NNNNNN.bin clean frames.")

    # inspect
    sp_i = sub.add_parser("inspect", help="Task-1 demo: GT + frame mapping.")
    _add_paths(sp_i)
    sp_i.set_defaults(func=cmd_inspect)

    # run
    sp_r = sub.add_parser("run", help="Full detect -> track -> metrics -> tables/plots.")
    _add_paths(sp_r)
    sp_r.add_argument("--out", default="track_eval", help="Output directory.")
    sp_r.add_argument("--cfg", default=None, help="OpenPCDet PointPillars cfg yaml.")
    sp_r.add_argument("--ckpt", default=None, help="PointPillars checkpoint .pth.")
    sp_r.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    sp_r.add_argument("--score-thresh", type=float, default=DEFAULT_SCORE_THRESH)
    sp_r.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    sp_r.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE)
    sp_r.add_argument("--iou-thresh", type=float, default=DEFAULT_IOU_THRESH)
    sp_r.add_argument("--gate", type=float, default=DEFAULT_GATE_M)
    sp_r.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    sp_r.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

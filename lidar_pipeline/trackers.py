"""
trackers.py

Task 3 — the three multi-object trackers run on PointPillars detections
(clean vs POPA). All share one interface so the evaluation can run them
interchangeably on the same detections:

  * ``KalmanFilterTracker`` — constant-velocity Kalman filter on the box centre,
    Hungarian association by centre distance. The lightweight baseline.

  * ``AB3DMOT`` — the "3D MOT baseline" (Weng et al.): a 10-D Kalman filter
    [x, y, z, theta, l, w, h, vx, vy, vz] with Hungarian association by
    bird's-eye-view IoU, and birth/death via ``min_hits`` / ``max_age``.

  * ``CenterPointTracker`` — CenterPoint's velocity-based greedy closest-point
    matcher (Yin et al.): predict each track forward by its last displacement,
    then greedily match detections to tracks by centre distance under a
    per-class gate. No Kalman smoothing.

Each tracker consumes a per-frame detection list (see
``detector_io.to_frame_list``) and produces a tidy DataFrame:
    frame, track_id, x, y, z, vx, vy, speed, dx, dy, dz, yaw, score, cls
with velocities in m/s (KITTI runs at 10 Hz, dt = 0.1 s) — giving the Track ID,
Object Position, Estimated Velocity and Trajectory the PDF asks for.

Only numpy + scipy are required. ``shapely`` is used for rotated-box IoU when
available, otherwise an axis-aligned BEV IoU fallback is used.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DT: float = 0.1  # 10 Hz

TRACK_COLUMNS = [
    "frame", "track_id", "x", "y", "z", "vx", "vy", "speed",
    "dx", "dy", "dz", "yaw", "score", "cls",
]

# Per-class centre-distance gates (metres) for CenterPoint-style matching.
CENTERPOINT_DIST_GATE = {
    "Car": 4.0, "Van": 4.0, "Truck": 4.0,
    "Pedestrian": 1.0, "Cyclist": 1.6,
    "_default": 3.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

try:  # rotated BEV IoU via shapely (available on Colab)
    from shapely.geometry import Polygon  # type: ignore
    _HAVE_SHAPELY = True
except Exception:  # pragma: no cover - local fallback
    _HAVE_SHAPELY = False


def _bev_corners(box: np.ndarray) -> np.ndarray:
    """4 BEV corners (x,y) of a box [x,y,z,dx,dy,dz,yaw]."""
    x, y, _, dx, dy, _, yaw = box[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = dx / 2.0, dy / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    rot = np.array([[c, -s], [s, c]])
    return (local @ rot.T) + np.array([x, y])


def bev_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Bird's-eye-view IoU of two 3D boxes [x,y,z,dx,dy,dz,yaw]."""
    if _HAVE_SHAPELY:
        pa, pb = Polygon(_bev_corners(a)), Polygon(_bev_corners(b))
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.area + pb.area - inter
        return float(inter / union) if union > 0 else 0.0
    # Axis-aligned fallback (ignores yaw).
    ax1, ay1 = a[0] - a[3] / 2, a[1] - a[4] / 2
    ax2, ay2 = a[0] + a[3] / 2, a[1] + a[4] / 2
    bx1, by1 = b[0] - b[3] / 2, b[1] - b[4] / 2
    bx2, by2 = b[0] + b[3] / 2, b[1] + b[4] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return float(inter / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Constant-velocity Kalman filter (velocity on x,y,z only)
# ─────────────────────────────────────────────────────────────────────────────

class _CVKalman:
    """Constant-velocity KF tracking a d-dim measurement plus (vx,vy,vz).

    State = [m_0 .. m_{d-1}, vx, vy, vz]; only the first 3 measurement dims
    (x, y, z) have associated velocity. Positions advance by v * dt each step.
    """

    def __init__(self, meas: np.ndarray, dt: float = DT):
        d = len(meas)
        self.d = d
        self.dt = dt
        n = d + 3
        self.x = np.zeros((n, 1))
        self.x[:d, 0] = meas
        self.F = np.eye(n)
        for k in range(3):  # x,y,z += v*dt
            self.F[k, d + k] = dt
        self.H = np.zeros((d, n))
        self.H[:d, :d] = np.eye(d)
        self.P = np.eye(n) * 10.0
        self.P[d:, d:] *= 100.0          # high initial velocity uncertainty
        self.Q = np.eye(n) * 0.01
        self.Q[d:, d:] *= 0.1
        self.R = np.eye(d) * 0.1

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas: np.ndarray) -> None:
        z = np.asarray(meas, dtype=float).reshape(self.d, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

    @property
    def meas(self) -> np.ndarray:
        return self.x[:self.d, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.d:self.d + 3, 0].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Track bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

class _Track:
    _next_id = 0

    def __init__(self, box: np.ndarray, score: float, cls: str, kf_dim: int):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.cls = cls
        self.score = score
        self.box = box.copy()                    # [x,y,z,dx,dy,dz,yaw]
        self.kf_dim = kf_dim
        if kf_dim == 7:
            meas = np.array([box[0], box[1], box[2], box[6], box[3], box[4], box[5]])
        elif kf_dim == 3:
            meas = box[:3].copy()
        else:  # 0 -> no Kalman
            meas = None
        self.kf = _CVKalman(meas) if meas is not None else None
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.last_center = box[:3].copy()
        self.vel = np.zeros(3)                   # m/s (for non-KF trackers)

    @classmethod
    def reset_ids(cls) -> None:
        cls._next_id = 0

    def predict(self) -> None:
        if self.kf is not None:
            self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, box: np.ndarray, score: float) -> None:
        prev_center = self.box[:3].copy()
        self.box = box.copy()
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        if self.kf is not None:
            if self.kf_dim == 7:
                meas = np.array([box[0], box[1], box[2], box[6], box[3], box[4], box[5]])
            else:
                meas = box[:3].copy()
            self.kf.update(meas)
        self.vel = (box[:3] - prev_center) / DT
        self.last_center = box[:3].copy()

    def state_box(self) -> np.ndarray:
        """Current best box estimate [x,y,z,dx,dy,dz,yaw]."""
        if self.kf is not None and self.kf_dim == 7:
            m = self.kf.meas  # [x,y,z,theta,l,w,h]
            return np.array([m[0], m[1], m[2], m[4], m[5], m[6], m[3]])
        if self.kf is not None and self.kf_dim == 3:
            c = self.kf.meas
            return np.array([c[0], c[1], c[2], self.box[3], self.box[4], self.box[5], self.box[6]])
        return self.box.copy()

    def report_velocity(self) -> np.ndarray:
        """(vx, vy) in m/s for the output row."""
        if self.kf is not None:
            return self.kf.velocity[:2]
        return self.vel[:2]


# ─────────────────────────────────────────────────────────────────────────────
# Base tracker
# ─────────────────────────────────────────────────────────────────────────────

class BaseTracker:
    """Common track lifecycle; subclasses implement ``_associate``."""

    name = "base"
    kf_dim = 3

    def __init__(self, max_age: int = 2, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[_Track] = []
        _Track.reset_ids()

    def _associate(self, boxes: np.ndarray, classes: list[str]
                   ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        raise NotImplementedError

    def step(self, frame: int, boxes: np.ndarray, scores: np.ndarray,
             classes: list[str]) -> list[dict]:
        for t in self.tracks:
            t.predict()

        matches, unmatched_det, _ = self._associate(boxes, classes)

        for det_i, trk_i in matches:
            self.tracks[trk_i].update(boxes[det_i], float(scores[det_i]))

        for det_i in unmatched_det:
            self.tracks.append(_Track(boxes[det_i], float(scores[det_i]),
                                      classes[det_i], self.kf_dim))

        # Kill stale tracks.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Emit confirmed tracks seen this frame.
        rows = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or frame < self.min_hits):
                box = t.state_box()
                vx, vy = t.report_velocity()
                rows.append({
                    "frame": frame, "track_id": t.id,
                    "x": box[0], "y": box[1], "z": box[2],
                    "vx": vx, "vy": vy, "speed": float(np.hypot(vx, vy)),
                    "dx": box[3], "dy": box[4], "dz": box[5], "yaw": box[6],
                    "score": t.score, "cls": t.cls,
                })
        return rows

    def run(self, frame_list: list[dict]) -> pd.DataFrame:
        self.tracks = []
        _Track.reset_ids()
        all_rows: list[dict] = []
        for fr in frame_list:
            all_rows.extend(self.step(fr["frame"], fr["boxes"],
                                      fr["scores"], fr["classes"]))
        if not all_rows:
            return pd.DataFrame(columns=TRACK_COLUMNS)
        return pd.DataFrame(all_rows, columns=TRACK_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Kalman filter tracker — centre-distance association
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilterTracker(BaseTracker):
    name = "kalman"
    kf_dim = 3

    def __init__(self, max_age: int = 2, min_hits: int = 3, dist_gate: float = 4.0):
        super().__init__(max_age, min_hits)
        self.dist_gate = dist_gate

    def _associate(self, boxes, classes):
        n_det, n_trk = len(boxes), len(self.tracks)
        if n_det == 0 or n_trk == 0:
            return [], list(range(n_det)), list(range(n_trk))
        cost = np.zeros((n_det, n_trk))
        for i in range(n_det):
            for j, t in enumerate(self.tracks):
                cost[i, j] = np.linalg.norm(boxes[i][:3] - t.state_box()[:3])
        det_idx, trk_idx = linear_sum_assignment(cost)
        matches, um_det, um_trk = [], [], []
        matched_d, matched_t = set(det_idx), set(trk_idx)
        um_det = [i for i in range(n_det) if i not in matched_d]
        um_trk = [j for j in range(n_trk) if j not in matched_t]
        for i, j in zip(det_idx, trk_idx):
            if cost[i, j] > self.dist_gate:
                um_det.append(i); um_trk.append(j)
            else:
                matches.append((i, j))
        return matches, um_det, um_trk


# ─────────────────────────────────────────────────────────────────────────────
# 2) AB3DMOT — 3D Kalman + BEV-IoU association
# ─────────────────────────────────────────────────────────────────────────────

class AB3DMOT(BaseTracker):
    name = "ab3dmot"
    kf_dim = 7

    def __init__(self, max_age: int = 2, min_hits: int = 3, iou_gate: float = 0.01):
        super().__init__(max_age, min_hits)
        self.iou_gate = iou_gate

    def _associate(self, boxes, classes):
        n_det, n_trk = len(boxes), len(self.tracks)
        if n_det == 0 or n_trk == 0:
            return [], list(range(n_det)), list(range(n_trk))
        iou = np.zeros((n_det, n_trk))
        for i in range(n_det):
            for j, t in enumerate(self.tracks):
                iou[i, j] = bev_iou(boxes[i], t.state_box())
        det_idx, trk_idx = linear_sum_assignment(-iou)  # maximise IoU
        matches, um_det, um_trk = [], [], []
        matched_d, matched_t = set(det_idx), set(trk_idx)
        um_det = [i for i in range(n_det) if i not in matched_d]
        um_trk = [j for j in range(n_trk) if j not in matched_t]
        for i, j in zip(det_idx, trk_idx):
            if iou[i, j] < self.iou_gate:
                um_det.append(i); um_trk.append(j)
            else:
                matches.append((i, j))
        return matches, um_det, um_trk


# ─────────────────────────────────────────────────────────────────────────────
# 3) CenterPoint tracker — velocity-based greedy closest-point
# ─────────────────────────────────────────────────────────────────────────────

class CenterPointTracker(BaseTracker):
    name = "centerpoint"
    kf_dim = 0  # no Kalman smoothing; uses last-displacement velocity

    def __init__(self, max_age: int = 3, min_hits: int = 1,
                 dist_gate: dict | None = None):
        super().__init__(max_age, min_hits)
        self.dist_gate = dist_gate or CENTERPOINT_DIST_GATE

    def _associate(self, boxes, classes):
        n_det, n_trk = len(boxes), len(self.tracks)
        if n_det == 0 or n_trk == 0:
            return [], list(range(n_det)), list(range(n_trk))
        # Predict each track forward by its last per-frame displacement.
        preds = np.array([t.last_center + t.vel * DT for t in self.tracks])
        cost = np.full((n_det, n_trk), 1e6)
        for i in range(n_det):
            for j, t in enumerate(self.tracks):
                if classes[i] != t.cls:      # match same-class only
                    continue
                cost[i, j] = np.linalg.norm(boxes[i][:3] - preds[j])
        # Greedy closest-first under a per-detection (per-class) gate.
        used_det, used_trk, matches = set(), set(), []
        pairs = sorted((cost[i, j], i, j)
                       for i in range(n_det) for j in range(n_trk))
        for c, i, j in pairs:
            gate = self.dist_gate.get(classes[i], self.dist_gate["_default"])
            if c > gate or i in used_det or j in used_trk:
                continue
            matches.append((i, j)); used_det.add(i); used_trk.add(j)
        um_det = [i for i in range(n_det) if i not in used_det]
        um_trk = [j for j in range(n_trk) if j not in used_trk]
        return matches, um_det, um_trk


# Registry so the evaluation can iterate over all three trackers.
TRACKERS = {
    "kalman": KalmanFilterTracker,
    "ab3dmot": AB3DMOT,
    "centerpoint": CenterPointTracker,
}


def build_tracker(name: str, **kwargs) -> BaseTracker:
    if name not in TRACKERS:
        raise KeyError(f"Unknown tracker '{name}'. Options: {list(TRACKERS)}")
    return TRACKERS[name](**kwargs)


if __name__ == "__main__":
    from lidar_pipeline.detector_io import make_synthetic_detections, to_frame_list

    dets = make_synthetic_detections(n_frames=15, drop_frames=(7,), noise=0.0)
    frames = to_frame_list(dets, frame_range=range(15))
    for name in TRACKERS:
        res = build_tracker(name).run(frames)
        n_ids = res["track_id"].nunique() if not res.empty else 0
        print(f"\n[{name}] {len(res)} rows, {n_ids} unique track ids")
        print(res.head(4).to_string(index=False))

"""
tracking_metrics.py

Task 3 — Step 3/4: metrics comparing tracking on clean vs POPA-attacked data,
against the KITTI ground truth (velodyne frame, from ``tracking_gt``).

Four metrics from the spec:

  Metric 1 — Velocity Error
      |V_estimated - V_groundtruth| per matched (frame, object). Mean and max.

  Metric 2 — ID Switches
      Number of times a ground-truth object's matched tracker ID changes over
      the frames it is tracked. Total + average per sequence.

  Metric 3 — Trajectory Error
      TE = mean over matched frames of sqrt((x-x̂)^2 + (y-ŷ)^2). Mean and max (m).

  Metric 4 — Track Fragmentation
      Number of times a ground-truth track is interrupted (matched -> lost ->
      matched again). Total.

Tracker hypotheses are associated to GT per frame by Hungarian matching on
centre distance under a gate, so the metrics follow the standard CLEAR-MOT
bookkeeping.

Reporting:
  * ``evaluate(tracked, gt)``            -> metrics for one (tracker, condition, sequence)
  * ``comparative_table(clean, popa)``   -> per-tracker clean-vs-POPA table
  * ``table1_tracking_performance(res)`` -> Table 1, aggregated over all sequences
  * ``table2_sequence_wise(res)``        -> Table 2, one row per (sequence, tracker, condition)
where ``res`` is nested: res[seq][tracker] = {"clean": eval_dict, "popa": eval_dict}.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Association: tracked hypotheses <-> ground truth (per frame)
# ─────────────────────────────────────────────────────────────────────────────

MATCH_COLUMNS = ["frame", "gt_id", "track_id", "dist", "gt_speed", "trk_speed",
                 "gt_x", "gt_y", "trk_x", "trk_y"]


def match_to_gt(tracked: pd.DataFrame, gt: pd.DataFrame,
                dist_gate: float = 2.0) -> pd.DataFrame:
    """Per-frame Hungarian matching of tracked boxes to GT by centre distance.

    Returns matched pairs with columns ``MATCH_COLUMNS``.
    """
    rows = []
    if tracked.empty or gt.empty:
        return pd.DataFrame(rows, columns=MATCH_COLUMNS)

    frames = sorted(set(gt["frame"]).union(set(tracked["frame"])))
    for f in frames:
        g = gt[gt["frame"] == f]
        t = tracked[tracked["frame"] == f]
        if g.empty or t.empty:
            continue
        gp = g[["x", "y"]].to_numpy()
        tp = t[["x", "y"]].to_numpy()
        cost = np.linalg.norm(gp[:, None, :] - tp[None, :, :], axis=2)
        gi, ti = linear_sum_assignment(cost)
        for a, b in zip(gi, ti):
            if cost[a, b] > dist_gate:
                continue
            grow, trow = g.iloc[a], t.iloc[b]
            rows.append({
                "frame": int(f),
                "gt_id": int(grow["track_id"]),
                "track_id": int(trow["track_id"]),
                "dist": float(cost[a, b]),
                "gt_speed": float(grow["speed"]),
                "trk_speed": float(trow["speed"]),
                "gt_x": float(grow["x"]), "gt_y": float(grow["y"]),
                "trk_x": float(trow["x"]), "trk_y": float(trow["y"]),
            })
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Velocity error
# ─────────────────────────────────────────────────────────────────────────────

def velocity_error(matches: pd.DataFrame) -> dict:
    if matches.empty:
        return {"mean_velocity_error": float("nan"), "max_velocity_error": float("nan")}
    err = (matches["trk_speed"] - matches["gt_speed"]).abs()
    return {"mean_velocity_error": float(err.mean()),
            "max_velocity_error": float(err.max())}


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — ID switches
# ─────────────────────────────────────────────────────────────────────────────

def id_switches(matches: pd.DataFrame) -> dict:
    """Count changes of the matched tracker id for each GT object over time."""
    total = 0
    per_gt = {}
    for gt_id, grp in matches.sort_values("frame").groupby("gt_id"):
        ids = grp["track_id"].to_numpy()
        switches = int(np.sum(ids[1:] != ids[:-1]))
        per_gt[int(gt_id)] = switches
        total += switches
    return {"total_id_switches": total, "per_gt": per_gt}


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — Trajectory error
# ─────────────────────────────────────────────────────────────────────────────

def trajectory_error(matches: pd.DataFrame) -> dict:
    if matches.empty:
        return {"mean_trajectory_error": float("nan"), "max_trajectory_error": float("nan")}
    dev = np.hypot(matches["trk_x"] - matches["gt_x"],
                   matches["trk_y"] - matches["gt_y"])
    return {"mean_trajectory_error": float(dev.mean()),
            "max_trajectory_error": float(dev.max())}


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4 — Track fragmentation
# ─────────────────────────────────────────────────────────────────────────────

def fragmentation(matches: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Count interruptions in each GT track's matched coverage.

    A fragmentation is a transition matched -> missing -> matched again, within
    the span of frames the GT object actually exists.
    """
    total_frag = 0
    per_gt = {}
    for gt_id, g in gt.groupby("track_id"):
        gframes = sorted(g["frame"].tolist())
        if not gframes:
            continue
        matched_frames = set(matches[matches["gt_id"] == gt_id]["frame"].tolist())
        seen_match = False
        in_gap = False
        frags = 0
        for f in gframes:
            if f in matched_frames:
                if in_gap and seen_match:
                    frags += 1        # resumed after a break
                seen_match = True
                in_gap = False
            elif seen_match:
                in_gap = True
        per_gt[int(gt_id)] = frags
        total_frag += frags
    return {"total_fragmentations": total_frag, "per_gt": per_gt}


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation for one (tracker, condition, sequence)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(tracked: pd.DataFrame, gt: pd.DataFrame,
             dist_gate: float = 2.0) -> dict:
    """Compute all four metrics for one (tracked, gt) pair."""
    matches = match_to_gt(tracked, gt, dist_gate=dist_gate)
    ve = velocity_error(matches)
    ids = id_switches(matches)
    te = trajectory_error(matches)
    fm = fragmentation(matches, gt)
    return {
        "matches": matches,
        "mean_velocity_error": ve["mean_velocity_error"],
        "max_velocity_error": ve["max_velocity_error"],
        "total_id_switches": ids["total_id_switches"],
        "mean_trajectory_error": te["mean_trajectory_error"],
        "max_trajectory_error": te["max_trajectory_error"],
        "total_fragmentations": fm["total_fragmentations"],
        "n_matched_boxes": len(matches),
        "n_gt_frames": int(gt["frame"].nunique()) if not gt.empty else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — Tracking Performance (aggregated over all sequences)
# ─────────────────────────────────────────────────────────────────────────────

def _pool(res: dict, tracker: str, condition: str) -> dict:
    """Aggregate one tracker/condition across all sequences.

    Errors are pooled over all matched rows (not a mean of per-sequence means);
    ID switches / fragmentations are summed; ``avg_id_switches_per_sequence`` is
    total / number-of-sequences.
    """
    all_matches, tot_idsw, tot_frag, n_seq = [], 0, 0, 0
    for seq, trackers in res.items():
        if tracker not in trackers:
            continue
        e = trackers[tracker][condition]
        all_matches.append(e["matches"])
        tot_idsw += e["total_id_switches"]
        tot_frag += e["total_fragmentations"]
        n_seq += 1
    m = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame(columns=MATCH_COLUMNS)
    ve = velocity_error(m)
    te = trajectory_error(m)
    return {
        "mean_velocity_error": ve["mean_velocity_error"],
        "max_velocity_error": ve["max_velocity_error"],
        "total_id_switches": tot_idsw,
        "avg_id_switches_per_sequence": float(tot_idsw / n_seq) if n_seq else 0.0,
        "mean_trajectory_error": te["mean_trajectory_error"],
        "max_trajectory_error": te["max_trajectory_error"],
        "total_fragmentations": tot_frag,
        "n_matched_boxes": len(m),
        "n_sequences": n_seq,
    }


def comparative_table(clean_eval: dict, popa_eval: dict) -> pd.DataFrame:
    """Per-tracker clean-vs-POPA table (works on either evaluate() or _pool() dicts)."""
    metrics = [
        ("mean_velocity_error", "Mean Velocity Error (m/s)"),
        ("max_velocity_error", "Max Velocity Error (m/s)"),
        ("total_id_switches", "Total ID Switches"),
        ("mean_trajectory_error", "Mean Trajectory Error (m)"),
        ("max_trajectory_error", "Max Trajectory Error (m)"),
        ("total_fragmentations", "Total Fragmentations"),
    ]
    rows = []
    for key, label in metrics:
        c, p = clean_eval.get(key, float("nan")), popa_eval.get(key, float("nan"))
        rows.append({"Metric": label, "Clean": c, "POPA": p, "Delta (POPA-Clean)": p - c})
    return pd.DataFrame(rows)


def table1_tracking_performance(res: dict, trackers: list[str] | None = None) -> pd.DataFrame:
    """Table 1 — one row per tracker, clean vs POPA for each metric, aggregated
    over all sequences present in ``res``."""
    if trackers is None:
        trackers = sorted({t for s in res.values() for t in s})
    rows = []
    for tr in trackers:
        c = _pool(res, tr, "clean")
        p = _pool(res, tr, "popa")
        rows.append({
            "Tracker": tr,
            "MeanVelErr_Clean": round(c["mean_velocity_error"], 3),
            "MeanVelErr_POPA": round(p["mean_velocity_error"], 3),
            "IDSw_Clean": c["total_id_switches"],
            "IDSw_POPA": p["total_id_switches"],
            "AvgIDSw/Seq_POPA": round(p["avg_id_switches_per_sequence"], 3),
            "TrajErr_Clean": round(c["mean_trajectory_error"], 3),
            "TrajErr_POPA": round(p["mean_trajectory_error"], 3),
            "Frag_Clean": c["total_fragmentations"],
            "Frag_POPA": p["total_fragmentations"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — Sequence-wise Results
# ─────────────────────────────────────────────────────────────────────────────

def table2_sequence_wise(res: dict) -> pd.DataFrame:
    """Table 2 — one row per (sequence, tracker, condition)."""
    rows = []
    for seq in sorted(res.keys()):
        for tracker in sorted(res[seq].keys()):
            for cond in ("clean", "popa"):
                e = res[seq][tracker][cond]
                rows.append({
                    "Sequence": seq, "Tracker": tracker, "Condition": cond.upper(),
                    "MeanVelErr": round(e["mean_velocity_error"], 3),
                    "MaxVelErr": round(e["max_velocity_error"], 3),
                    "IDSwitches": e["total_id_switches"],
                    "MeanTrajErr": round(e["mean_trajectory_error"], 3),
                    "MaxTrajErr": round(e["max_trajectory_error"], 3),
                    "Fragmentations": e["total_fragmentations"],
                    "MatchedBoxes": e["n_matched_boxes"],
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: synthetic clean vs POPA detections built from REAL GT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from lidar_pipeline.tracking_gt import load_gt_tracks
    from lidar_pipeline.detector_io import DETECTION_COLUMNS, to_frame_list
    from lidar_pipeline.trackers import TRACKERS, build_tracker

    def gt_to_detections(gt: pd.DataFrame, noise: float, seed: int) -> pd.DataFrame:
        """Fabricate detections from GT: clean=noise 0; POPA=jitter+drops on
        attack-burst frames (idx%6<3), to exercise the metrics realistically."""
        rng = np.random.default_rng(seed)
        rows = []
        for _, r in gt.iterrows():
            f = int(r["frame"])
            attacked = (f % 6) < 3
            jit = np.zeros(2)
            if noise > 0 and attacked:
                jit = rng.normal(0, noise, 2)
                if rng.random() < 0.25:
                    continue  # dropped detection -> fragmentation / ID switch
            rows.append({"frame": f, "x": r["x"] + jit[0], "y": r["y"] + jit[1],
                         "z": r["z"], "dx": r["l"], "dy": r["w"], "dz": r["h"],
                         "yaw": r["ry"], "score": 0.9, "cls": r["cls"]})
        return pd.DataFrame(rows, columns=DETECTION_COLUMNS)

    seq = "0000"
    gt = load_gt_tracks(seq=seq)
    fr = range(int(gt["frame"].min()), int(gt["frame"].max()) + 1)
    clean_det = gt_to_detections(gt, noise=0.0, seed=1)
    adv_det = gt_to_detections(gt, noise=0.8, seed=2)

    res = {seq: {}}
    for name in TRACKERS:
        ct = build_tracker(name).run(to_frame_list(clean_det, fr))
        at = build_tracker(name).run(to_frame_list(adv_det, fr))
        res[seq][name] = {"clean": evaluate(ct, gt), "popa": evaluate(at, gt)}

    print("=== Table 1 — Tracking Performance (aggregate) ===")
    print(table1_tracking_performance(res).to_string(index=False))
    print("\n=== Table 2 — Sequence-wise Results ===")
    print(table2_sequence_wise(res).to_string(index=False))

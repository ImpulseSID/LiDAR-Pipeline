"""
tracking_plots.py

Task 3 — the four required visualizations for the clean-vs-POPA comparison:
  1. Clean vs POPA trajectory plots (BEV x-y, tracked vs ground truth)
  2. Velocity-over-time plots
  3. ID-switch examples
  4. Track-fragmentation examples

All plots use matplotlib with the headless 'Agg' backend (works on Colab and
in CI with no display) and are saved as PNGs under ``outputs/``. Filenames are
namespaced by sequence and tracker, e.g. ``trajectories_0000_ab3dmot.png``.
"""

from __future__ import annotations

import sys
import matplotlib
matplotlib.use("Agg")  # headless (Colab / no display)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import PROJECT_ROOT

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"


def _ensure_out(path: Union[str, Path, None]) -> Path:
    out = Path(path) if path is not None else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tag(seq: str, tracker: str) -> str:
    return f"{seq}_{tracker}"


# ─────────────────────────────────────────────────────────────────────────────
# 1) Clean vs POPA trajectories (BEV)
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(clean_tracked: pd.DataFrame, popa_tracked: pd.DataFrame,
                      gt: pd.DataFrame, seq: str, tracker: str,
                      out_dir: Union[str, Path, None] = None) -> Path:
    """Bird's-eye trajectory overlay: GT vs clean-tracked vs POPA-tracked."""
    out = _ensure_out(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    gt_min = gt["track_id"].min() if not gt.empty else None
    for ax, tracked, title in [
        (axes[0], clean_tracked, f"{tracker} — CLEAN (seq {seq})"),
        (axes[1], popa_tracked, f"{tracker} — POPA (seq {seq})"),
    ]:
        for gid, g in gt.groupby("track_id"):
            g = g.sort_values("frame")
            ax.plot(g["x"], g["y"], "--", color="0.6", linewidth=1.2,
                    label="GT" if gid == gt_min else None)
        if not tracked.empty:
            for tid, t in tracked.groupby("track_id"):
                t = t.sort_values("frame")
                ax.plot(t["x"], t["y"], "-", linewidth=1.0, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("x (m, forward)")
        ax.set_ylabel("y (m, left)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Trajectories: Clean vs POPA  (seq {seq}, {tracker})")
    fig.tight_layout()
    path = out / f"trajectories_{_tag(seq, tracker)}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 2) Velocity over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_velocity_over_time(clean_matches: pd.DataFrame, popa_matches: pd.DataFrame,
                            gt: pd.DataFrame, seq: str, tracker: str,
                            out_dir: Union[str, Path, None] = None,
                            max_objects: int = 4) -> Path:
    """Estimated speed vs GT speed over time, for the longest-lived GT objects."""
    out = _ensure_out(out_dir)
    # Pick the GT tracks with the most frames (most informative).
    order = gt.groupby("track_id").size().sort_values(ascending=False)
    gt_ids = list(order.index[:max_objects])
    n = max(len(gt_ids), 1)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), squeeze=False)

    for row, gid in enumerate(gt_ids):
        ax = axes[row][0]
        g = gt[gt["track_id"] == gid].sort_values("frame")
        ax.plot(g["frame"], g["speed"], "k--", linewidth=1.4, label="GT speed")
        for m, colour, lbl in [(clean_matches, "tab:blue", "clean est."),
                               (popa_matches, "tab:red", "POPA est.")]:
            mm = m[m["gt_id"] == gid].sort_values("frame")
            if not mm.empty:
                ax.plot(mm["frame"], mm["trk_speed"], "-o", markersize=3,
                        color=colour, label=lbl)
        ax.set_title(f"GT track {gid} ({g['cls'].iloc[0]})")
        ax.set_xlabel("frame"); ax.set_ylabel("speed (m/s)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f"Velocity over time (seq {seq}, {tracker})")
    fig.tight_layout()
    path = out / f"velocity_over_time_{_tag(seq, tracker)}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3) ID-switch examples
# ─────────────────────────────────────────────────────────────────────────────

def plot_id_switches(popa_matches: pd.DataFrame, seq: str, tracker: str,
                     out_dir: Union[str, Path, None] = None) -> Path:
    """For each GT object, show the assigned tracker id per frame under POPA;
    a change in the id (colour) marks an ID switch."""
    out = _ensure_out(out_dir)
    fig, ax = plt.subplots(figsize=(11, 5))
    gt_ids = sorted(popa_matches["gt_id"].unique()) if not popa_matches.empty else []

    for row, gid in enumerate(gt_ids):
        mm = popa_matches[popa_matches["gt_id"] == gid].sort_values("frame")
        ax.scatter(mm["frame"], [row] * len(mm), c=mm["track_id"],
                   cmap="tab20", s=55, edgecolors="k", linewidths=0.4)
        ids = mm["track_id"].to_numpy()
        frames = mm["frame"].to_numpy()
        for i in range(1, len(ids)):
            if ids[i] != ids[i - 1]:
                ax.annotate("switch", (frames[i], row),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=7, color="red", ha="center")
    ax.set_yticks(range(len(gt_ids)))
    ax.set_yticklabels([f"GT {g}" for g in gt_ids])
    ax.set_xlabel("frame")
    ax.set_title(f"Assigned tracker ID per GT object under POPA (seq {seq}, {tracker})\n"
                 "colour = tracker id; 'switch' marks an identity change")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = out / f"id_switches_{_tag(seq, tracker)}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4) Fragmentation examples
# ─────────────────────────────────────────────────────────────────────────────

def plot_fragmentation(clean_matches: pd.DataFrame, popa_matches: pd.DataFrame,
                       gt: pd.DataFrame, seq: str, tracker: str,
                       out_dir: Union[str, Path, None] = None) -> Path:
    """Timeline of which frames each GT object was successfully tracked; grey
    gaps under POPA are the fragmentations."""
    out = _ensure_out(out_dir)
    gt_ids = sorted(gt["track_id"].unique())
    fig, ax = plt.subplots(figsize=(11, 0.9 * max(len(gt_ids), 1) + 2))

    for row, gid in enumerate(gt_ids):
        g_frames = sorted(gt[gt["track_id"] == gid]["frame"].tolist())
        clean_f = set(clean_matches[clean_matches["gt_id"] == gid]["frame"].tolist())
        popa_f = set(popa_matches[popa_matches["gt_id"] == gid]["frame"].tolist())
        for f in g_frames:
            ax.scatter(f, row + 0.15, marker="s", s=22,
                       color="tab:blue" if f in clean_f else "0.85")
            ax.scatter(f, row - 0.15, marker="s", s=22,
                       color="tab:red" if f in popa_f else "0.85")
    ax.set_yticks(range(len(gt_ids)))
    ax.set_yticklabels([f"GT {g}" for g in gt_ids])
    ax.set_xlabel("frame")
    ax.set_title(f"Track coverage timeline (seq {seq}, {tracker})\n"
                 "top (blue)=clean tracked, bottom (red)=POPA tracked; grey = lost (fragmentation)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = out / f"fragmentation_{_tag(seq, tracker)}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_all_plots(clean_tracked, popa_tracked, clean_matches, popa_matches,
                   gt, seq: str, tracker: str, out_dir=None) -> list[Path]:
    """Generate all four figures for one (sequence, tracker); returns paths."""
    return [
        plot_trajectories(clean_tracked, popa_tracked, gt, seq, tracker, out_dir),
        plot_velocity_over_time(clean_matches, popa_matches, gt, seq, tracker, out_dir),
        plot_id_switches(popa_matches, seq, tracker, out_dir),
        plot_fragmentation(clean_matches, popa_matches, gt, seq, tracker, out_dir),
    ]


if __name__ == "__main__":
    # Render all four plots from real-GT-derived synthetic data as a self-test.
    import numpy as np
    from lidar_pipeline.tracking_gt import load_gt_tracks
    from lidar_pipeline.detector_io import DETECTION_COLUMNS, to_frame_list
    from lidar_pipeline.trackers import build_tracker
    from lidar_pipeline.tracking_metrics import evaluate

    seq = "0000"
    gt = load_gt_tracks(seq=seq)
    fr = range(int(gt["frame"].min()), int(gt["frame"].max()) + 1)

    def mk(noise, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for _, r in gt.iterrows():
            f = int(r["frame"]); att = (f % 6) < 3
            jx = rng.normal(0, noise) if noise and att else 0.0
            jy = rng.normal(0, noise) if noise and att else 0.0
            rows.append({"frame": f, "x": r["x"] + jx, "y": r["y"] + jy, "z": r["z"],
                         "dx": r["l"], "dy": r["w"], "dz": r["h"], "yaw": r["ry"],
                         "score": 0.9, "cls": r["cls"]})
        return pd.DataFrame(rows, columns=DETECTION_COLUMNS)

    clean, adv = mk(0.0, 1), mk(0.8, 2)
    tr = "ab3dmot"
    ct = build_tracker(tr).run(to_frame_list(clean, fr))
    at = build_tracker(tr).run(to_frame_list(adv, fr))
    ce, pe = evaluate(ct, gt), evaluate(at, gt)
    paths = make_all_plots(ct, at, ce["matches"], pe["matches"], gt, seq, tr)
    print("Wrote:")
    for p in paths:
        print(f"  {p}  ({p.stat().st_size} bytes)")

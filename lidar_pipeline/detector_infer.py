"""
detector_infer.py

Task 3 — PointPillars detector wrapper (runs on Colab GPU via OpenPCDet).

This is the ONLY heavyweight / GPU step. It loads a pretrained KITTI
PointPillars model through OpenPCDet, runs it over a folder of ``.bin`` LiDAR
frames, and writes the detections as a plain CSV (schema from ``detector_io``).
The tracking + metrics + plotting steps then run anywhere with no GPU.

IMPORTANT: all ``pcdet`` / ``torch`` imports are done *inside* the functions so
this module imports cleanly locally (where those packages are absent). Only
``run_detection`` actually needs them, and only on Colab.

OpenPCDet KITTI PointPillars produces, per frame:
    pred_boxes  : (M,7) [x, y, z, dx, dy, dz, heading]  in the LiDAR frame
    pred_scores : (M,)
    pred_labels : (M,)  1-indexed into class_names = [Car, Pedestrian, Cyclist]
which maps directly onto the detection CSV schema.

Typical Colab usage (see the setup notebook):
    from lidar_pipeline.detector_infer import run_sequence_detections
    run_sequence_detections(
        seq="0000",
        clean_dir="/content/clean_frames/0000",
        adv_dir="/content/adv_frames/0000",
        out_dir="outputs", cfg_file=CFG, ckpt=CKPT,
    )
"""

from __future__ import annotations

import sys
import glob
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.detector_io import DETECTION_COLUMNS, save_detections, detections_path

# KITTI PointPillars class order (pred_labels are 1-indexed into this).
DEFAULT_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _frame_number(file_path: str) -> int:
    """Extract the integer frame number from a filename stem.

    Handles KITTI ``000042.bin`` and adversarial ``adv_0042.bin`` alike, so
    clean and adversarial detections share the same frame index space.
    """
    stem = Path(file_path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def _build_demo_dataset(frames_dir: Path, cfg, class_names, logger, ext: str = ".bin"):
    """Minimal OpenPCDet dataset over a flat folder of .bin frames.

    Imported lazily; mirrors OpenPCDet's tools/demo.py DemoDataset.
    """
    from pcdet.datasets import DatasetTemplate  # type: ignore

    class _DemoDataset(DatasetTemplate):
        def __init__(self):
            super().__init__(
                dataset_cfg=cfg.DATA_CONFIG, class_names=class_names,
                training=False, root_path=frames_dir, logger=logger,
            )
            self.sample_file_list = sorted(glob.glob(str(frames_dir / f"*{ext}")))

        def __len__(self):
            return len(self.sample_file_list)

        def __getitem__(self, index):
            points = np.fromfile(self.sample_file_list[index],
                                 dtype=np.float32).reshape(-1, 4)
            data_dict = self.prepare_data(
                data_dict={"points": points, "frame_id": index})
            return data_dict

    return _DemoDataset()


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(
    frames_dir: Union[str, Path],
    output_csv: Union[str, Path],
    cfg_file: Union[str, Path],
    ckpt: Union[str, Path],
    class_names: list[str] | None = None,
    score_thresh: float = 0.1,
    ext: str = ".bin",
) -> pd.DataFrame:
    """Run PointPillars over ``frames_dir`` and write detections to ``output_csv``.

    Parameters
    ----------
    frames_dir  : folder of ``.bin`` LiDAR frames (clean or adversarial).
    output_csv  : destination CSV (detector_io schema).
    cfg_file    : OpenPCDet config yaml (e.g. pointpillar.yaml).
    ckpt        : pretrained checkpoint (.pth).
    class_names : detection classes; defaults to KITTI [Car, Pedestrian, Cyclist].
    score_thresh: drop detections below this confidence.

    Returns the detections DataFrame (also written to disk).
    """
    # ---- lazy heavy imports (Colab / GPU only) --------------------------
    import torch  # type: ignore
    from pcdet.config import cfg, cfg_from_yaml_file  # type: ignore
    from pcdet.models import build_network, load_data_to_gpu  # type: ignore
    from pcdet.utils import common_utils  # type: ignore

    frames_dir = Path(frames_dir)
    class_names = class_names or DEFAULT_CLASS_NAMES
    logger = common_utils.create_logger()

    cfg_from_yaml_file(str(cfg_file), cfg)
    dataset = _build_demo_dataset(frames_dir, cfg, class_names, logger, ext=ext)
    logger.info(f"Detecting over {len(dataset)} frames in {frames_dir}")

    model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names),
                          dataset=dataset)
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            data_dict = dataset[idx]
            frame_no = _frame_number(dataset.sample_file_list[idx])
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()      # (M,7)
            scores = pred_dicts[0]["pred_scores"].cpu().numpy()    # (M,)
            labels = pred_dicts[0]["pred_labels"].cpu().numpy()    # (M,) 1-indexed

            for b, s, lab in zip(boxes, scores, labels):
                if s < score_thresh:
                    continue
                cls = (class_names[int(lab) - 1]
                       if 1 <= int(lab) <= len(class_names) else "Unknown")
                rows.append({
                    "frame": frame_no,
                    "x": float(b[0]), "y": float(b[1]), "z": float(b[2]),
                    "dx": float(b[3]), "dy": float(b[4]), "dz": float(b[5]),
                    "yaw": float(b[6]), "score": float(s), "cls": cls,
                })

    df = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    save_detections(df, output_csv)
    logger.info(f"Wrote {len(df)} detections to {output_csv}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence convenience (clean + adversarial in one call)
# ─────────────────────────────────────────────────────────────────────────────

def run_sequence_detections(
    seq: str,
    clean_dir: Union[str, Path],
    adv_dir: Union[str, Path],
    out_dir: Union[str, Path],
    cfg_file: Union[str, Path],
    ckpt: Union[str, Path],
    class_names: list[str] | None = None,
    score_thresh: float = 0.1,
) -> tuple[Path, Path]:
    """Detect on both conditions for one sequence, writing
    ``clean_detections_<seq>.csv`` and ``adv_detections_<seq>.csv``.

    Returns the two output paths.
    """
    clean_csv = detections_path(out_dir, seq, "clean")
    adv_csv = detections_path(out_dir, seq, "adv")
    run_detection(clean_dir, clean_csv, cfg_file, ckpt, class_names, score_thresh)
    run_detection(adv_dir, adv_csv, cfg_file, ckpt, class_names, score_thresh)
    return clean_csv, adv_csv


if __name__ == "__main__":
    # Local import check only — real inference requires torch + pcdet on Colab.
    print("detector_infer imported OK (torch/pcdet are imported lazily at call time).")
    print("Detection CSV schema:", DETECTION_COLUMNS)
    print("Frame-number parsing:",
          {"000042.bin": _frame_number("000042.bin"),
           "adv_0042.bin": _frame_number("adv_0042.bin")})

# Task 3 — POPA Attack Impact on Object Tracking

Evaluate how the **POPA** adversarial attack degrades 3‑D multi‑object tracking
versus clean data. The pipeline runs a **PointPillars** detector (OpenPCDet,
pretrained KITTI) on clean and POPA‑attacked LiDAR frames, tracks objects with an
**AB3DMOT‑style 3‑D Kalman tracker**, and computes four metrics — velocity error,
ID switches, trajectory error, track fragmentation — with clean‑vs‑POPA
comparison tables and plots.

---

## Contents
1. [What it produces](#what-it-produces)
2. [Environment setup](#environment-setup-arch--conda)
3. [Build OpenPCDet](#build-openpcdet)
4. [Get the checkpoint](#get-the-checkpoint)
5. [Generate adversarial frames](#generate-adversarial-frames)
6. [Run the evaluation](#run-the-evaluation)
7. [CLI reference](#cli-reference)

---

## What it produces
Per sequence, under the `--out` directory:
- `table1_aggregate.csv` / `.md` — clean vs POPA, aggregate over all sequences.
- `table2_sequence_wise.csv` / `.md` — per‑sequence metrics.
- `<seq>/clean_detections.csv`, `<seq>/adv_detections.csv` — PointPillars detections.
- `<seq>/clean_tracking_results.csv`, `<seq>/adv_tracking_results.csv` — tracker output.
- `<seq>/seq<seq>_trajectories.png`, `_velocity_over_time.png`, `_id_switches.png`, `_fragmentation.png`.

The metrics (computed for both clean and POPA):
- **Velocity Error** `|v_est − v_gt|` (mean / max, m/s) — GT velocity is a finite
  difference of GT box centres at 10 Hz (`dt = 0.1 s`), in the sensor frame.
- **ID Switches** — times a GT track's associated tracker id changes (total / avg).
- **Trajectory Error** — mean/max BEV distance between GT and matched track (m).
- **Track Fragmentation** — times a GT track's coverage is interrupted.

---


## Environment setup

Use a **conda** environment that carries its own CUDA 12.4 toolkit and a
compatible compiler, independent of the system CUDA/gcc.

### 1. Install Miniforge
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate
```

### 2. Create + activate the environment
```bash
conda create -y -n task3 -c conda-forge -c nvidia python=3.12 cuda-toolkit=12.4 gxx=12
conda activate task3
```

### 3. Install Python packages
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "numpy<2" pandas matplotlib scipy tabulate \
            pyquaternion easydict SharedArray tensorboardX kornia scikit-image
pip install "opencv-python-headless==4.9.0.80"
pip install spconv-cu118 cumm-cu118
```

### 4. Toolchain + cumm header env
Set these in the **same shell** you build and run in:
```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
# cumm JIT-builds its core_cc on first import and needs its bundled headers:
export CPATH="$CONDA_PREFIX/lib/python3.12/site-packages/cumm/include:$CPATH"
export CPLUS_INCLUDE_PATH="$CPATH"

nvcc --version   # must read "release 12.4" (conda's), NOT the system CUDA 13.x
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
```

---

## Build OpenPCDet
```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
pip install -e . --no-build-isolation

sed -i '/Argo2Dataset/d' pcdet/datasets/__init__.py
cd ..


python -c "from pcdet.ops.iou3d_nms import iou3d_nms_utils; print('ops OK')"
python -c "import torch,spconv,cumm,pcdet; print('OK', spconv.__version__, cumm.__version__)"
```

---

## Get the checkpoint
```bash
pip install gdown
gdown 1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm -O ~/pointpillar_7728.pth   # ~19 MB
```

---

## Generate adversarial frames
POPA frames come from the Module‑4 attack (numpy‑only, CPU).

**Single sequence** (flat output — `adv_0000.bin … adv_0153.bin`, attacked +
clean‑copy cooldown frames):
```bash
cd <repo-root>
python -m lidar_pipeline.attack --seq 0000 --start 0 --count 154 \
       --core-fraction 0.12 --n-flicker 6 --out ./adv_frames
```

**Multiple sequences** — pass several ids (or `all`) to `--seq`. With more than
one sequence the attack writes per‑sequence subfolders `<out>/<seq>/` (with a
single sequence it stays flat), which is exactly what the evaluator reads with
`--adv-dir ./adv_frames --seq all`.

```bash
cd <repo-root>

# a specific subset
python -m lidar_pipeline.attack --seq 0000 0001 0002 --start 0 --count 100000 \
       --core-fraction 0.12 --n-flicker 6 --out ./adv_frames

# every available sequence, all frames
python -m lidar_pipeline.attack --seq all --start 0 --count 100000 \
       --core-fraction 0.12 --n-flicker 6 --out ./adv_frames
```
`--count` is clamped to each sequence's length, so a large value (e.g. `100000`)
simply means "all frames of every sequence". Output layout:
`./adv_frames/0000/adv_0000.bin`, `./adv_frames/0001/…`, etc.

Then evaluate them all at once with `--adv-dir <repo>/adv_frames --seq all`
(Table 2 gets a row per sequence; Table 1 aggregates them).

---

## Attack success metrics (optional)

A separate Module‑4 check ([`attack_metric.py`](lidar_pipeline/attack_metric.py))
reports how much the attack degrades the *detector* on the target vehicle —
3‑D IoU and confidence drop, with an attack success rate (IoU < 0.55 AND
drop > 0.75). It runs on CPU (no PointPillars) and is independent of the tracking
evaluation. `--adv-dir` accepts a flat folder or per‑seq subfolders, and `--seq`
takes one id, several, or `all`.

Single sequence (flat `adv_frames`):
```bash
cd <repo-root>
python -m lidar_pipeline.attack_metric --seq 0000 --adv-dir ./adv_frames
```

Multiple / all sequences (per‑seq `adv_frames/<seq>/` layout) — prints a table +
summary per sequence, then an overall aggregate:
```bash
python -m lidar_pipeline.attack_metric --seq 0000 0001 0002 --adv-dir ./adv_frames
python -m lidar_pipeline.attack_metric --seq all --adv-dir ./adv_frames
```
Quick check on a few pairs: add `--max-pairs 12`.

---

## Run the evaluation
The detector config's `_BASE_CONFIG_` resolves relative to `OpenPCDet/tools`, so
run from there with the repo on `PYTHONPATH`:
```bash
cd OpenPCDet/tools
PYTHONPATH=<repo> python -m lidar_pipeline.tracking_evaluation run \
    --seq 0000 \
    --adv-dir <repo>/adv_frames \
    --out <repo>/track_eval \
    --cfg cfgs/kitti_models/pointpillar.yaml \
    --ckpt ~/pointpillar_7728.pth \
    --score-thresh 0.3
```
- One sequence: `--seq 0000`. Several: `--seq 0000 0001 0002`. All: `--seq all`.
- `--adv-dir` accepts a flat folder (`adv_XXXX.bin`) or per‑seq subfolders
  (`<adv-dir>/<seq>/adv_XXXX.bin`) — the per‑seq form is used when present.
- `--out` can be any path (relative to the current dir or absolute).

Task‑1 sanity check (GT boxes in LiDAR coords + adv/clean frame mapping; no GPU
model needed):
```bash
python -m lidar_pipeline.tracking_evaluation inspect --seq 0000 --adv-dir <repo>/adv_frames
```

Concrete example (paths as verified on the RTX 3050 box):
```bash
cd ~/Github/LiDAR-Pipeline/OpenPCDet/tools
PYTHONPATH=$HOME/Github/LiDAR-Pipeline python -m lidar_pipeline.tracking_evaluation run \
    --seq 0000 --adv-dir $HOME/Github/LiDAR-Pipeline/adv_frames \
    --out $HOME/Github/LiDAR-Pipeline/track_eval \
    --cfg cfgs/kitti_models/pointpillar.yaml --ckpt $HOME/pointpillar_7728.pth --score-thresh 0.3
```

---

## CLI reference

`run` subcommand:

| Flag | Default | Description |
|------|---------|-------------|
| `--seq` | `0000` | Sequence id(s); space‑separated list, or `all`. |
| `--adv-dir` | `data/adversarial` | Adversarial frames dir (flat or per‑seq). |
| `--out` | `track_eval` | Output directory for CSVs/tables/plots. |
| `--cfg` | env `POINTPILLARS_CFG` | OpenPCDet PointPillars config yaml. |
| `--ckpt` | env `POINTPILLARS_CKPT` | PointPillars checkpoint `.pth`. |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--score-thresh` | `0.3` | Detection score threshold. |
| `--min-hits` | `3` | Tracker: frames before a track is confirmed. |
| `--max-age` | `2` | Tracker: unmatched frames before deletion. |
| `--iou-thresh` | `0.01` | Min BEV IoU to accept an association. |
| `--gate` | `3.0` | Max centre distance (m) for association. |
| `--no-plots` | off | Skip PNG plots. |
| `--label-dir`, `--calib-dir`, `--velodyne-root` | dataset defaults | Path overrides. |


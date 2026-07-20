# Module 4 — POPA Attack & Metrics

Adversarial LiDAR generation and evaluation for the `lidar_pipeline` project.

This module takes real KITTI LiDAR frames, applies the **POPA** (Partial Object
Persistence) attack to the vehicles in each frame, and measures how badly the
attack degrades 3-D object detection.

- `attack.py` — generates adversarial LiDAR frames (`adv_XXXX.bin`).
- `attack_metric.py` — scores original vs adversarial frames (IoU + Confidence Drop).

---

## Table of Contents

1. [What is POPA?](#what-is-popa)
2. [The physical rationale (why the visible face matters)](#the-physical-rationale)
3. [How the attack works](#how-the-attack-works)
4. [How the metrics work](#how-the-metrics-work)
5. [Typical workflow](#typical-workflow)
6. [`attack.py` CLI reference](#attackpy-cli-reference)
7. [`attack_metric.py` CLI reference](#attack_metricpy-cli-reference)
8. [Tuning guide (controlling ASR)](#tuning-guide)
9. [Public API](#public-api)
10. [Caveats & limitations](#caveats--limitations)
11. [Dependencies](#dependencies)

---

## What is POPA?

**POPA = Partial Object Persistence.** The attacked object never disappears
completely — instead, *different parts of it vanish and reappear across
consecutive frames*. A small persistent core is always present, while the rest
of the object flickers in and out on a temporal schedule.

The goals:

- **Partially preserve the object** — it is never fully gone.
- **Remove different regions per frame** — the shape keeps changing.
- **Destabilise 3-D object detection** — the bounding box jumps and shrinks.
- **Confuse multi-object tracking** — ID switches and broken tracks.
- **Maintain a realistic LiDAR appearance** — only real points are removed, and
  only from vehicles; the rest of the scene is untouched.

### Attack cadence (burst / cooldown)

The attack runs in **bursts** over the real frame sequence: it attacks
`N` consecutive frames, then **cools down** (leaves frames untouched) for `N`
frames, then attacks the next `N`, and so on. Default is **attack 3, cool down 3**:

```
attack 000000–000002  →  cool 000003–000005  →  attack 000006–000008  →  cool ...
```

Output frames are named after the **source frame number**:

```
000000.bin → adv_0000.bin      000006.bin → adv_0006.bin
```

Cooldown frames are written as **clean copies** by default, so the output folder
is a complete, continuous sequence (useful for tracking evaluation).

---

## The physical rationale

A LiDAR is line-of-sight: it only captures the surfaces of a vehicle that face
the sensor and are not self-occluded. The far side and far end of a vehicle are
never seen, so the object in the point cloud is already a **partial shell**
(typically an "L" shape), not a full box.

Which face is visible depends on pose:

- **Lead vehicle** (ahead, same direction): you see its **rear** (+ near side).
- **Oncoming vehicle**: you see its **front**.
- **Crossing / adjacent vehicle**: you see a **side** (+ nearest end).

**Consequence for the attack:** you can only remove points that actually exist —
the visible, sensor-facing surface. So instead of splitting the vehicle by an
arbitrary axis (front/side/rear), the attack partitions each vehicle by
**distance from the sensor** into range shells and operates on those real points.

---

## How the attack works

For each **attacked** frame:

1. **Detect vehicles.** The frame is voxelized, semantically segmented, and
   instance-segmented (DBSCAN). Every vehicle instance is extracted. Non-vehicle
   points (ground, buildings, vegetation) are never modified.

2. **Partition each vehicle by sensor range.** Points are ordered by distance
   from the sensor and split into shells (nearest → farthest):

   ```
   core , f0 , f1 , f2 , ...
   ```

   - `core` = nearest `--core-fraction` of the points (default **10%**) — the
     most reliably visible surface. **Always kept** (the object persists).
   - `f0 … f{n-1}` = the remaining points split into `--n-flicker` thin shells
     (default **10** shells of ~9% each).

3. **Choose which shells survive this frame.** Within a burst, the visible set
   depends on the frame's position:

   ```
   burst pos 0 → keep  core                (~10%)
   burst pos 1 → keep  core + f0            (~19%)
   burst pos 2 → keep  core + f1            (~19%)
   ```

   So different fragments persist across the burst — this is the POPA flicker.

4. **Remove the missing shells.** A raw point is deleted only if it is **inside
   the vehicle's bounding box** *and* its **distance from the sensor** falls in a
   missing shell. Everything else in the scene is preserved.

5. **Write** `adv_XXXX.bin` in KITTI binary layout (flat `float32` `[x,y,z,i,…]`).

**Cooldown** frames are copied through unchanged.

### Kept fraction ≈ how much of each vehicle survives

```
kept_fraction ≈ core_fraction + (1 − core_fraction) / n_flicker
```

Since the metric's confidence drop is `1 − kept_fraction`, keeping the kept
fraction **below ~0.25** is what clears the `Drop > 0.75` success bar.

---

## How the metrics work

`attack_metric.py` pairs each `adv_NNNN.bin` with its original `NNNNNN.bin`,
runs the same simulated detector on both, and computes two numbers for the
**target vehicle** (the largest vehicle in the original frame).

### IoU (3-D bounding-box overlap)

Axis-aligned 3-D IoU between the target's box in the original frame and its
matched box in the adversarial frame. The adversarial object is matched to the
original by best IoU (falling back to nearest centre within a gate). Lower IoU =
the detector's box moved/shrank more.

```
IoU = intersection_volume / union_volume
```

### Confidence Drop

Confidence is normalised by the target object's **own** full point count (not a
fixed cap), so the drop reflects how much of *that* object was destroyed:

```
conf_original     = 1.0
conf_adversarial  = surviving_points / original_points
confidence_drop   = conf_original − conf_adversarial
```

If the object becomes undetectable in the adversarial frame, `conf_adversarial = 0`
and the drop is `1.0`.

### Success criterion

An attacked frame is a **SUCCESS** only if **both** hold:

```
IoU            < 0.55
Confidence Drop > 0.75
```

**ASR** (Attack Success Rate) = fraction of attacked pairs that are a success.
Cooldown/clean frames (adversarial file is byte-identical to the original) are
detected cheaply and **excluded** from the stats by default, since there is no
attack to measure.

---

## Typical workflow

```powershell
# 1. Generate adversarial frames for the first 60 dataset frames
python -m lidar_pipeline.attack --start 0 --count 60 --out data/adversarial

# 2. Score original vs adversarial
python -m lidar_pipeline.attack_metric --adv-dir data/adversarial
```

Quick sanity check on a small batch (the detector is compute-heavy):

```powershell
python -m lidar_pipeline.attack --count 12
python -m lidar_pipeline.attack_metric --max-pairs 12
```

> Both scripts can be run either as a module (`python -m lidar_pipeline.attack`)
> or directly (`python .\lidar_pipeline\attack.py`). Run from the project root.

---

## `attack.py` CLI reference

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `--start` | `0` | `0` – `N-1` (N = total frames in dataset) | 0-based index of the first dataset frame to process. |
| `--count` | `12` | `≥ 1` | Number of consecutive frames to process from `--start`. |
| `--attack-burst` | `3` | `≥ 1` | Consecutive frames attacked per cycle. |
| `--cooldown` | `3` | `≥ 0` | Consecutive untouched frames after each burst. |
| `--voxel-size` | `0.1` | `> 0.0` (typical: `0.05`–`0.5`) | Voxel size (m) for the detection pre-pass. Larger = coarser/faster. |
| `--core-fraction` | `0.10` | `0.0`–`1.0` (typical: `0.05`–`0.20`) | Nearest fraction of each vehicle kept as the persistent core. **Higher → keeps more → lower drop → lower ASR.** |
| `--n-flicker` | `10` | `≥ 1` (typical: `5`–`15`) | Number of thin shells the rest of the points split into. **Higher → thinner fragments → higher drop → higher ASR.** |
| `--skip-cooldown` | off | flag (no value) | Only write attacked frames (skip clean cooldown copies). |
| `--out` | `data/adversarial` | any valid directory path | Output directory for `adv_XXXX.bin`. |

The source frame directory is fixed to `data/velodyne/training/velodyne`.

**Examples**

```powershell
# Attack 3 / cool down 3, aggressive (default)
python -m lidar_pipeline.attack --count 60

# Land ASR in the 75–90% band (fragments near the drop threshold)
python -m lidar_pipeline.attack --count 60 --core-fraction 0.11 --n-flicker 7

# Longer bursts, only write attacked frames
python -m lidar_pipeline.attack --count 100 --attack-burst 5 --cooldown 2 --skip-cooldown
```

---

## `attack_metric.py` CLI reference

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `--orig-dir` | `data/velodyne/training/velodyne` | any valid directory path | Directory of original `NNNNNN.bin` frames. |
| `--adv-dir` | `data/adversarial` | any valid directory path | Directory of adversarial `adv_NNNN.bin` frames. |
| `--voxel-size` | `0.1` | `> 0.0` (typical: `0.05`–`0.5`) | Voxel size for the metric's detector (keep it equal to the attack's). |
| `--max-pairs` | all | `≥ 1` | Evaluate only the first N pairs (handy for a quick check). |
| `--include-clean` | off | flag (no value) | Also include untouched cooldown frames (IoU=1, drop=0) in the stats. |

**Output columns:** `#`, `Adversarial`, `IoU`, `ConfOrig`, `ConfAdv`, `Drop`,
`Result` (`SUCCESS` / `fail` / `clean`). A `!` next to `IoU` or `Drop` marks a
condition that was **not** met. The summary reports mean IoU, mean drop, how many
frames met each condition, and the overall ASR.

---

## Tuning guide

Both `--core-fraction` and `--n-flicker` set the surviving fraction per vehicle:

```
kept_fraction ≈ core_fraction + (1 − core_fraction) / n_flicker
drop          ≈ 1 − kept_fraction     (success needs drop > 0.75, i.e. kept < 0.25)
```

| Goal | Settings | Approx. kept | Approx. ASR |
|------|----------|--------------|-------------|
| Maximum impact | `--core-fraction 0.10 --n-flicker 10` | ~0.19 | ~100% |
| Realistic 75–90% | `--core-fraction 0.11 --n-flicker 7` | ~0.24 | ~75–90% |
| Milder (~2/3) | `--core-fraction 0.13 --n-flicker 6` | ~0.27 | ~65% |

> **Note on quantization:** because a 3-frame burst uses only 3 visibility
> patterns (`core`, `core+f0`, `core+f1`), ASR tends to snap toward multiples of
> 1/3 (≈33/67/100%). Landing precisely in 75–90% means putting the `core+shell`
> frames right at the `drop = 0.75` boundary, so detector/clustering noise
> splits them between pass and fail.

---

## Public API

### `attack.py`

- `extract_all_vehicles(frame, voxel_size, region_names, core_fraction, sensor_origin, min_points)`
  → list of per-vehicle dicts (`object`, `bbox_min/max`, `regions`, `range_edges`, `info`).
- `generate_adversarial_sequence(frame_paths, output_dir, attack_burst, cooldown, voxel_size, region_names, persistent, core_fraction, sensor_origin, prefix, pad_m, min_points, write_cooldown, random_seed, verbose)`
  → list of written paths. The main driver.
- `popa_attack_frame(original_frame, vehicles, kept_regions, region_names, sensor_origin, pad_m)`
  → adversarial `(M, 4)` array for one frame.
- `build_popa_schedule(frame_indices, region_names, persistent, attack_burst, cooldown)`
  → `{idx: {"mode", "kept"}}`.
- `popa_regions_for_frame(frame_index, region_names, persistent, attack_burst, cooldown)`
  → list of visible shell names for that frame.
- `is_attack_frame(frame_index, attack_burst, cooldown)` → bool.
- `extract_target_and_regions(...)` — legacy single-object, axis-based split; kept
  for the other submodules (`stealth_attack`, `tracking_attack`, `adaptive_*`).

Key constants: `CORE_FRACTION=0.10`, `N_FLICKER_SHELLS=10`, `ATTACK_BURST=3`,
`COOLDOWN=3`, `BBOX_PAD_M=0.15`, `SENSOR_ORIGIN=[0,0,1.73]`.

### `attack_metric.py`

- `detect_vehicles(frame, voxel_size)` → list of detection dicts.
- `compute_3d_iou(min_a, max_a, min_b, max_b)` → float.
- `evaluate_pair(orig_path, adv_path, voxel_size)` → metrics dict (or `None` if no
  target vehicle; `clean=True` for untouched cooldown frames).
- `discover_pairs(orig_dir, adv_dir)` → list of `(orig, adv)` path pairs.

Key constants: `IOU_SUCCESS_MAX=0.55`, `CONF_DROP_SUCCESS_MIN=0.75`,
`VOXEL_SIZE=0.1`, `MATCH_GATE_M=3.0`.

---

## Caveats & limitations

- **Proxy detector.** The "detector" is the project's segmentation pipeline
  (voxelize → semantic → DBSCAN), not a trained 3-D detector. It is a loose
  heuristic that can label many clusters per frame as "vehicle".
- **Confidence is a point-retention proxy**, not a neural network's objectness
  score. Because the attack directly controls point count, `Drop > 0.75` is
  close to "removed > 75% of points" — so a very high ASR is partly *by
  construction*, not an independent result. Treat 100% ASR as a sign the metric
  is easy, not proof of a strong real-world attack.
- **IoU collapse is genuine** — keeping a small nearest fragment really does
  shrink the 3-D box, which would hurt a real detector too.
- For a rigorous evaluation, plug in a real pretrained detector (e.g.
  PointPillars / CenterPoint via OpenPCDet) and read its actual confidence.
- With a small `--core-fraction`, small/distant vehicles may keep too few points
  to be re-detected. The points still exist in the cloud (POPA holds), but the
  detector loses the box — which shows up as a near-1.0 drop.

---

## Dependencies

```
pip install numpy
```

`attack.py` and `attack_metric.py` need only **NumPy** plus the project's own
`lidar_pipeline` modules (`loader`, `voxelization`, `semantic_segmentation`,
`instance_segmentation`). `scipy` and `matplotlib` are only required by the
separate `tracking_evaluation.py`, not by this module.

Input frames: KITTI Velodyne `.bin` files in `data/velodyne/training/velodyne`.

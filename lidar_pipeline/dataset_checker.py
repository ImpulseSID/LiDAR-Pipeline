"""
dataset_checker.py

Scans every .bin file in the KITTI velodyne dataset and checks for:
  - Corrupted / unreadable files
  - Empty files (0 bytes or 0 points)
  - Wrong dimensions (size not a multiple of 16 bytes → not valid (N, 4) float32)
  - Suspiciously sparse frames (< 100 points — likely corrupted)

"""

import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

# Ensure the project root is on sys.path so `lidar_pipeline` is importable
# when running: python lidar_pipeline/dataset_checker.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lidar_pipeline.loader import get_all_frames, default_velodyne_dir

# Constants
DATA_DIR = default_velodyne_dir()
MIN_POINT_THRESHOLD = 100    # frames with fewer points than this are flagged
BYTES_PER_POINT     = 16     # 4 × float32 = 4 × 4 bytes

_LINE = "-" * 55


# Result dataclass

@dataclass
class FrameResult:
    filename:   str
    status:     str  = ""     # "ok" | "empty" | "corrupt" | "bad_dim" | "sparse"
    num_points: int  = 0
    file_size:  int  = 0
    error_msg:  str  = ""


# Single-frame checker

def check_frame(filepath: Path) -> FrameResult:
    """
    Inspect a single .bin file for integrity issues.

    Checks:
      1. File is readable and exists
      2. File size > 0
      3. File size is a multiple of BYTES_PER_POINT (16)
      4. NumPy can parse it correctly
      5. Number of points >= MIN_POINT_THRESHOLD
    """
    result = FrameResult(filename=filepath.name)

    # Check 1: existence / readable
    try:
        result.file_size = filepath.stat().st_size
    except OSError as e:
        result.status    = "corrupt"
        result.error_msg = f"Cannot read file: {e}"
        return result

    # Check 2: empty file
    if result.file_size == 0:
        result.status    = "empty"
        result.error_msg = "File is 0 bytes"
        return result

    # Check 3: dimension validity (must be multiple of 16 bytes)
    if result.file_size % BYTES_PER_POINT != 0:
        result.status = "bad_dim"
        result.error_msg = (
            f"File size {result.file_size} bytes is not divisible by {BYTES_PER_POINT} "
            f"(remainder: {result.file_size % BYTES_PER_POINT})"
        )
        return result

    # Check 4: NumPy parse
    try:
        points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        result.num_points = len(points)
    except Exception as e:
        result.status    = "corrupt"
        result.error_msg = f"NumPy parse error: {e}"
        return result

    # Check 5: sparse frame
    if result.num_points < MIN_POINT_THRESHOLD:
        result.status    = "sparse"
        result.error_msg = f"Only {result.num_points} points (threshold: {MIN_POINT_THRESHOLD})"
        return result

    # All checks passed
    result.status = "ok"
    return result


# Full dataset checker

def check_dataset(data_dir: Path = DATA_DIR) -> list[FrameResult]:

    # Scan every .bin file in data_dir and return a list of FrameResult objects.

    frames = get_all_frames(data_dir)
    total  = len(frames)

    print(f"\nScanning {total:,} frames in: {data_dir}")
    print(_LINE)

    results: list[FrameResult] = []
    start_time = time.time()

    for i, fp in enumerate(frames, 1):
        result = check_frame(fp)
        results.append(result)

        # Progress indicator every 500 frames
        if i % 500 == 0 or i == total:
            elapsed = time.time() - start_time
            fps_rate = i / elapsed
            eta = (total - i) / fps_rate if fps_rate > 0 else 0
            print(
                f"  [{i:>5}/{total}]  {elapsed:5.1f}s elapsed  "
                f"ETA: {eta:4.1f}s  "
                f"-- speed: {fps_rate:.0f} frames/s",
                end="\r",
            )

    elapsed_total = time.time() - start_time
    print(f"\n  Done in {elapsed_total:.2f}s" + " " * 30)

    return results


# Reporting

def _status_icon(status: str) -> str:
    icons = {
        "ok":      "[OK]     ",
        "empty":   "[EMPTY]  ",
        "corrupt": "[CORRUPT]",
        "bad_dim": "[BAD DIM]",
        "sparse":  "[SPARSE] ",
    }
    return icons.get(status, f"? {status}")


def print_report(results: list[FrameResult]) -> None:
    # Lists all problematic frames with their issue descriptions.
    total    = len(results)
    ok       = sum(1 for r in results if r.status == "ok")
    empty    = sum(1 for r in results if r.status == "empty")
    corrupt  = sum(1 for r in results if r.status == "corrupt")
    bad_dim  = sum(1 for r in results if r.status == "bad_dim")
    sparse   = sum(1 for r in results if r.status == "sparse")
    problems = total - ok

    print()
    print("  KITTI Dataset Integrity Report")
    print("-" * 45)
    print(f"  Dataset path      : {DATA_DIR}")
    print(f"  Total frames      : {total:,}")
    print(f"  [OK]  Valid frames   : {ok:,}")
    print(f"  [!!]  Empty files    : {empty}")
    print(f"  [!!]  Corrupted      : {corrupt}")
    print(f"  [??]  Wrong dimensions: {bad_dim}")
    print(f"  [??]  Sparse frames  : {sparse}")
    print(_LINE)

    if problems == 0:
        print("  All frames passed! Dataset is clean. [OK]")
    else:
        print(f"  {problems} problem(s) found:")
        print()
        for r in results:
            if r.status != "ok":
                print(f"  {_status_icon(r.status)}  {r.filename:>15s}  --  {r.error_msg}")


    # Basic stats for valid frames
    valid_points = [r.num_points for r in results if r.status == "ok"]
    if valid_points:
        print(f"\n  Valid frame statistics:")
        print(f"    Min points / frame : {min(valid_points):,}")
        print(f"    Max points / frame : {max(valid_points):,}")
        print(f"    Avg points / frame : {int(np.mean(valid_points)):,}")
        print()


if __name__ == "__main__":
    print("Verify Dataset Integrity")
    print("_" * 45)

    try:
        results = check_dataset(DATA_DIR)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print_report(results)

#!/usr/bin/env python3
"""
Rotate quaternions in pose CSVs from Open3D convention (+Z forward) to ROS
convention (+X forward) by applying a −90° rotation about the Y-axis.

The transformation converts:
    +Z forward  →  +X forward
    +X right    →  +Y right
    +Y up       →  +Z up

Only the quaternion columns (``qx, qy, qz, qw``) are modified; all other columns
are passed through unchanged.

Usage
-----
    wur-av-bench convert-axes INPUT_DIR OUTPUT_DIR

Arguments
---------
INPUT_DIR   Directory containing source ``*.csv`` files with ``qx qy qz qw`` columns
            (comma or semicolon separated; delimiter is auto-detected).
OUTPUT_DIR  Destination directory for converted CSVs.

Author : Thomas Versmissen <thomas.versmissen@wur.nl>
License: MIT - see LICENSE for full text
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Need to rotate -90 degrees around Y-axis to go from Open3D (+Z forward) to ROS (+X forward) convention
_X_TO_Z_ROTATION = R.from_euler('y', -90, degrees=True)

quat_cols = ["qx", "qy", "qz", "qw"]

def _detect_delimiter(path: Path) -> str:
    counts = {",": 0, ";": 0}
    with path.open() as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            counts[","] += line.count(",")
            counts[";"] += line.count(";")
    return "," if counts[","] >= counts[";"] else ";"


def transform_coordinates_and_quaternions(df: pd.DataFrame) -> pd.DataFrame:
    df_t = df.copy()

    if all(c in df.columns for c in quat_cols):
        quats_old = df[quat_cols].to_numpy()            # Extract the quaternion data
        rot_old   = R.from_quat(quats_old)              # Convert to rotation objects
        rot_new   = rot_old * _X_TO_Z_ROTATION          # Apply our coordinate transformation
        df_t[quat_cols] = rot_new.as_quat()
    else:
        raise ValueError("Quaternion columns not found in the DataFrame")

    return df_t


def process_csv(path_in: Path, path_out: Path) -> int:
    """Convert *path_in* from +Z forward to +X forward and save to *path_out*. Returns row count."""

    sep = _detect_delimiter(path_in)
    df = pd.read_csv(path_in, sep=sep)
    
    print(f"  Original columns: {list(df.columns)}")

    # check if all cells have a valid float value and is not empty
    if df[quat_cols].isnull().any().any():
        raise ValueError(f"Missing values found in quaternion columns {quat_cols} of {path_in}")
        if not np.isfinite(df[quat_cols].to_numpy()).all():
            raise ValueError(f"Non-finite values found in quaternion columns {quat_cols} of {path_in}")


    df_transformed = transform_coordinates_and_quaternions(df)
    
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df_transformed.to_csv(path_out, index=False)
    
    return len(df_transformed)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Rotate quaternions in pose CSVs from +Z forward (Open3D) to +X forward (ROS) convention."
    )
    p.add_argument("input",  type=Path, help="Directory containing source *.csv files")
    p.add_argument("output", type=Path, help="Destination directory for converted CSVs")
    args = p.parse_args(argv)

    src = args.input.resolve()
    if not src.is_dir():
        sys.exit(f"Input directory does not exist: {src}")

    dst = args.output.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(src.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No .csv files found in {src}")

    print(f"Converting ground truth files from +Z forward to +X forward...")
    print(f"Input directory: {src}")
    print(f"Output directory: {dst}")
    print(f"Transformation: 90° rotation about Y-axis")
    print()

    total_rows = 0
    for f in csv_files:
        print(f"Processing {f.name}…")
        rows = process_csv(f, dst / f.name)
        total_rows += rows
        print(f"  ✓ saved → {dst / f.name}  ({rows} rows)")

    print(f"\nDone: {len(csv_files)} file(s), {total_rows} total rows → {dst}")
    print(f"Ground truth data now uses +X forward direction (ROS convention)")


if __name__ == "__main__":
    main()
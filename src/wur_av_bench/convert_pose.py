#!/usr/bin/env python3
"""
Convert plant-node annotations (edge + pt cylinder endpoints) into pose CSVs
(position + quaternion).

The direction vector ``pt − edge`` is converted to a quaternion that rotates ``+X``
onto that direction.  The edge point becomes the position (``x, y, z``).

Usage
-----
    wur-av-bench convert-pose INPUT_DIR [OUTPUT_DIR] [--quat-order {xyzw,wxyz}]

Arguments
---------
INPUT_DIR       Directory containing source ``*.csv`` files (comma or semicolon
                separated; delimiter is auto-detected).
OUTPUT_DIR      Destination directory for converted CSVs.
                Defaults to ``INPUT_DIR/poses`` when omitted.
--quat-order    Column order for the quaternion: ``xyzw`` (default) or ``wxyz``.

Output columns
--------------
``ID, x, y, z`` + quaternion columns, plus every other original column
(e.g. ``class``, ``visibility``).  The raw coordinate columns
(``x_edge, y_edge, z_edge, x_pt, y_pt, z_pt``) are removed.

Author : Thomas Versmissen <thomas.versmissen@wur.nl>
License: MIT - see LICENSE for full text
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

_REF = np.array([0.0, 0.0, 1.0])  # +X axis
_TOL = 1e-8                        # Zero-vector tolerance
_COORD_COLS: List[str] = [
    "x_edge", "y_edge", "z_edge",
    "x_pt", "y_pt", "z_pt",
]


def vector_to_quat(vec: np.ndarray) -> np.ndarray:  # (3,) → (4,)
    """Quaternion (x, y, z, w) rotating +X to *vec*; returns identity for zero vec."""
    norm = np.linalg.norm(vec)
    if norm < _TOL:
        return np.array([0.0, 0.0, 0.0, 1.0])
    target = vec / norm
    rot, _ = R.align_vectors([target], [_REF])
    return rot.as_quat()  # (x, y, z, w)


def _detect_delimiter(path: Path) -> str:
    """Detect comma vs semicolon in first few lines."""
    counts = {",": 0, ";": 0}
    with path.open() as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            counts[","] += line.count(",")
            counts[";"] += line.count(";")
    return "," if counts[","] >= counts[";"] else ";"


def process_csv(path_in: Path, path_out: Path, quat_order: str = "xyzw") -> int:
    """Convert *path_in* to pose format and save to *path_out*. Returns row count."""

    sep = _detect_delimiter(path_in)
    df = pd.read_csv(path_in, sep=sep)

    # Compute direction vectors pt − edge
    v = df[["x_pt", "y_pt", "z_pt"]].values - df[["x_edge", "y_edge", "z_edge"]].values
    quats = np.vstack([vector_to_quat(vec) for vec in v])  # (N, 4)

    if quat_order == "wxyz":
        quats = quats[:, [3, 0, 1, 2]]  # reorder → (w, x, y, z)
        qcols = ["qw", "qx", "qy", "qz"]
    else:
        qcols = ["qx", "qy", "qz", "qw"]

    pose = pd.DataFrame({
        "ID": df["ID"],
        "x": df["x_edge"],
        "y": df["y_edge"],
        "z": df["z_edge"],
        **{col: quats[:, i] for i, col in enumerate(qcols)},
    })

    # Keep every column **except** the raw coordinate columns we just used
    extras = df.drop(columns=[c for c in _COORD_COLS if c in df.columns] + ["ID"])
    out = pd.concat([pose, extras], axis=1)

    path_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path_out, index=False)
    return len(out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Convert node annotations (edge+pt) to pose CSVs (position + quaternion)."
    )
    p.add_argument("input",  type=Path, help="Directory containing source *.csv files")
    p.add_argument("output", type=Path, nargs="?", default=None,
                   help="Destination directory (default: <input>/poses)")
    p.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="xyzw",
                   help="Quaternion column order (default: xyzw)")
    args = p.parse_args(argv)

    src = args.input.resolve()
    if not src.is_dir():
        sys.exit(f"Input directory does not exist: {src}")

    dst = (args.output or src / "poses").resolve()
    dst.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(src.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No .csv files found in {src}")

    total_rows = 0
    for f in csv_files:
        print(f"Processing {f.name}…")
        rows = process_csv(f, dst / f.name, quat_order=args.quat_order)
        total_rows += rows
        print(f"  ✓ saved → {dst / f.name}  ({rows} rows)")

    print(f"Done: {len(csv_files)} file(s), {total_rows} total rows → {dst}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fit piecewise linear lines to plant stem edge keypoints.

Reads ground-truth CSVs (one per plant, output of ``wur-av-bench convert-axes``) and fits
three connected linear segments (bottom / center / top) along the z-axis using
constrained optimisation.  Continuity is enforced by construction:
    end of bottom == start of center,  end of center == start of top.

The fitted segment parameters are written to ``detection_filter_params.json`` as a list of
three cylinder dicts per plant (bottom / center / top):

    {
      "<plant_id>": [
        { "pointa": [x, y, z], "pointb": [x, y, z], "radius": 0.1 },
        ...
      ]
    }

Usage
-----
    wur-av-bench fit-stem-lines <gt_folder> [--output <dir>] [--radius <m>] [--extension <m>]

Arguments
---------
    gt_folder           Folder containing ground-truth CSV files (one per plant).
    --output <dir>      Output directory for results.  Default: ``fit_stem_lines_output``.
    --radius <m>        Cylinder radius written to the JSON (metres).  Default: ``0.1``.
    --extension <m>     Length added to each segment end for the cylinder endpoints (metres).
                        Default: ``0.04``.

Input CSV format
----------------
One CSV file per plant.  Only the position columns are required.

    x, y, z     Node position in metres

Author : Thomas A.W. Versmissen <thomas.versmissen@wur.nl>
License: MIT - see LICENSE for full text
Version: 2.0.0
Created: 2025-01-27
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _extend_segment(p0: np.ndarray, p1: np.ndarray,
                    extension_m: float) -> Tuple[List[float], List[float]]:
    """Return endpoints extended by *extension_m* on both sides."""
    v = np.asarray(p1) - np.asarray(p0)
    length = float(np.linalg.norm(v))
    if length < 1e-9:
        return list(map(float, p0)), list(map(float, p1))
    u = v / length
    return (
        (np.asarray(p0) - u * extension_m).astype(float).tolist(),
        (np.asarray(p1) + u * extension_m).astype(float).tolist(),
    )




def _detect_delimiter(path: Path) -> str:
    """Return the most common delimiter (',' or ';') found in the first 5 lines."""
    counts = {",": 0, ";": 0}
    with path.open() as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            counts[","] += line.count(",")
            counts[";"] += line.count(";")
    return "," if counts[","] >= counts[";"] else ";"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=_detect_delimiter(path))


def _filter_valid_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with finite x, y, z coordinates."""
    mask = (
        df['x'].notna() & df['y'].notna() & df['z'].notna() &
        np.isfinite(df['x']) & np.isfinite(df['y']) & np.isfinite(df['z'])
    )
    filtered = df[mask].copy()
    print(f"  Filtered {len(df)} total keypoints -> {len(filtered)} valid edge keypoints")
    return filtered


# ── Line fitting ──────────────────────────────────────────────────────────────

def _fit_line_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 3-D line using SVD.  Returns (centroid, unit_direction)."""
    if len(points) < 2:
        raise ValueError("Need at least 2 points to fit a line")
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    direction = Vt[0]
    if direction[2] < 0:
        direction = -direction
    return centroid, direction


def _divide_z_range(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Divide the z-range of *points* into three equal segments.

    Returns (min_z, boundary_1, boundary_2, max_z).
    """
    z = points[:, 2]
    min_z, max_z = float(z.min()), float(z.max())
    third = (max_z - min_z) / 3.0
    return min_z, min_z + third, min_z + 2 * third, max_z


def _fit_three_segments(bottom: np.ndarray, center: np.ndarray, top: np.ndarray,
                        min_z: float, z1: float, z2: float,
                        max_z: float) -> Dict[str, Any]:
    """Fit three connected line segments using constrained optimisation.

    Optimisation variables (8 scalars):
        bottom_start_xy, connection_1_xy, connection_2_xy, top_end_xy

    Continuity is enforced by construction (bottom.end == center.start, etc.).
    """
    all_pts = np.vstack([bottom, center, top])
    g_centroid, g_dir = _fit_line_svd(all_pts)

    def _proj_to_z(centroid: np.ndarray, direction: np.ndarray, z_target: float) -> np.ndarray:
        if abs(direction[2]) < 1e-10:
            idx = int(np.argmin(np.abs(all_pts[:, 2] - z_target)))
            return all_pts[idx, :2]
        t = (z_target - centroid[2]) / direction[2]
        return (centroid + t * direction)[:2]

    x0 = np.concatenate([
        _proj_to_z(g_centroid, g_dir, min_z),
        _proj_to_z(g_centroid, g_dir, z1),
        _proj_to_z(g_centroid, g_dir, z2),
        _proj_to_z(g_centroid, g_dir, max_z),
    ])

    def _unpack(x: np.ndarray):
        bs = np.array([x[0], x[1], min_z])
        c1 = np.array([x[2], x[3], z1])
        c2 = np.array([x[4], x[5], z2])
        te = np.array([x[6], x[7], max_z])
        segs = [
            (bs, c1, (c1 - bs) / np.linalg.norm(c1 - bs)),
            (c1, c2, (c2 - c1) / np.linalg.norm(c2 - c1)),
            (c2, te, (te - c2) / np.linalg.norm(te - c2)),
        ]
        return segs, c1, c2

    def _pt_to_seg_dist_sq(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        len_sq = float(np.dot(ab, ab))
        if len_sq < 1e-12:
            return float(np.dot(pt - a, pt - a))
        t = np.clip(np.dot(pt - a, ab) / len_sq, 0.0, 1.0)
        diff = pt - (a + t * ab)
        return float(np.dot(diff, diff))

    def _objective(x: np.ndarray) -> float:
        segs, _, _ = _unpack(x)
        pts_per_seg = [bottom, center, top]
        return sum(
            _pt_to_seg_dist_sq(pt, a, b)
            for (a, b, _), pts in zip(segs, pts_per_seg)
            for pt in pts
        )

    result = minimize(_objective, x0, method='BFGS',
                      options={'maxiter': 2000, 'gtol': 1e-5})
    if not result.success:
        print(f"    Warning: Optimisation did not fully converge ({result.message})")

    segs, conn1, conn2 = _unpack(result.x)
    (b_start, b_end, b_dir), (c_start, c_end, c_dir), (t_start, t_end, t_dir) = segs

    return {
        'bottom': (b_start, b_end, b_dir),
        'center': (c_start, c_end, c_dir),
        'top':    (t_start, t_end, t_dir),
        'connections': (conn1, conn2),
        'optimization_info': {
            'success': bool(result.success),
            'final_error': float(result.fun),
            'iterations': int(result.nit),
            'message': result.message,
        },
    }


def _fit_three_segment_lines(edge_points: np.ndarray) -> Dict[str, Any]:
    """Fit three connected segments to *edge_points* and return results dict."""
    if len(edge_points) < 6:
        raise ValueError(f"Need at least 6 points, got {len(edge_points)}")

    min_z, z1, z2, max_z = _divide_z_range(edge_points)
    z = edge_points[:, 2]

    print(f"    Z-range: {min_z:.3f} to {max_z:.3f}")
    print(f"    Segments: bottom [{min_z:.3f},{z1:.3f}], "
          f"center [{z1:.3f},{z2:.3f}], top [{z2:.3f},{max_z:.3f}]")

    bottom_pts = edge_points[z <= z1]
    center_pts = edge_points[(z > z1) & (z <= z2)]
    top_pts    = edge_points[z > z2]

    print(f"    Points per segment: bottom={len(bottom_pts)}, "
          f"center={len(center_pts)}, top={len(top_pts)}")

    # Fall back to equal-count split when a segment is empty
    if min(len(bottom_pts), len(center_pts), len(top_pts)) < 1:
        print("    Warning: empty segment detected — redistributing points equally")
        idx = np.argsort(z)
        n = len(edge_points) // 3
        bottom_pts = edge_points[idx[:n]]
        center_pts = edge_points[idx[n:2*n]]
        top_pts    = edge_points[idx[2*n:]]
        z1 = float(bottom_pts[:, 2].max())
        z2 = float(center_pts[:, 2].max())

    print("    Fitting with constrained optimisation...")
    fit = _fit_three_segments(bottom_pts, center_pts, top_pts, min_z, z1, z2, max_z)

    b_start, b_end, b_dir = fit['bottom']
    c_start, c_end, c_dir = fit['center']
    t_start, t_end, t_dir = fit['top']
    conn1, conn2 = fit['connections']

    results: Dict[str, Any] = {
        'z_boundaries': {
            'min_z': float(min_z), 'boundary_1': float(z1),
            'boundary_2': float(z2), 'max_z': float(max_z),
        },
        'point_counts': {
            'bottom': int(len(bottom_pts)), 'center': int(len(center_pts)),
            'top': int(len(top_pts)), 'total': int(len(edge_points)),
        },
        'segments': {
            'bottom': {
                'start_point': b_start.tolist(), 'end_point': b_end.tolist(),
                'centroid': ((b_start + b_end) / 2).tolist(),
                'direction': b_dir.tolist(), 'z_range': [float(min_z), float(z1)],
            },
            'center': {
                'start_point': c_start.tolist(), 'end_point': c_end.tolist(),
                'centroid': ((c_start + c_end) / 2).tolist(),
                'direction': c_dir.tolist(), 'z_range': [float(z1), float(z2)],
            },
            'top': {
                'start_point': t_start.tolist(), 'end_point': t_end.tolist(),
                'centroid': ((t_start + t_end) / 2).tolist(),
                'direction': t_dir.tolist(), 'z_range': [float(z2), float(max_z)],
            },
        },
        'connectivity_points': {
            'bottom_center': conn1.tolist(),
            'center_top':    conn2.tolist(),
        },
        'fitting_method': 'constrained_optimisation',
        'optimization_info': fit['optimization_info'],
    }

    results['quality_metrics'] = _calculate_quality(edge_points, results)
    return results


def _calculate_quality(points: np.ndarray, results: Dict[str, Any]) -> Dict[str, float]:
    """Return per-segment and overall RMS fit errors."""
    b = results['z_boundaries']
    z = points[:, 2]
    masks = {
        'bottom': z <= b['boundary_1'],
        'center': (z > b['boundary_1']) & (z <= b['boundary_2']),
        'top':    z > b['boundary_2'],
    }

    def _dist(pt: np.ndarray, a: np.ndarray, b_pt: np.ndarray) -> float:
        ab = b_pt - a
        len_sq = float(np.dot(ab, ab))
        if len_sq < 1e-10:
            return float(np.linalg.norm(pt - a))
        t = np.clip(np.dot(pt - a, ab) / len_sq, 0.0, 1.0)
        return float(np.linalg.norm(pt - (a + t * ab)))

    all_errors: List[float] = []
    seg_rms: Dict[str, float] = {}
    for name, mask in masks.items():
        seg = results['segments'][name]
        a = np.array(seg['start_point'])
        b_pt = np.array(seg['end_point'])
        errs = [_dist(pt, a, b_pt) for pt in points[mask]]
        seg_rms[f'{name}_rms_error'] = (
            float(np.sqrt(np.mean(np.array(errs) ** 2))) if errs else 0.0
        )
        all_errors.extend(errs)

    return {
        'overall_rms_error': float(np.sqrt(np.mean(np.array(all_errors) ** 2))) if all_errors else 0.0,
        'max_error':  float(np.max(all_errors))  if all_errors else 0.0,
        'mean_error': float(np.mean(all_errors)) if all_errors else 0.0,
        **seg_rms,
    }


# ── Per-plant processing ──────────────────────────────────────────────────────

def _process_plant(gt_file: Path) -> Optional[Dict[str, Any]]:
    """Fit lines to one plant CSV.  Returns results dict or None on failure."""
    try:
        print(f"Processing {gt_file.name}...")
        df = _read_csv(gt_file)
        edge_df = _filter_valid_keypoints(df)

        if len(edge_df) < 6:
            print(f"  Skipping: only {len(edge_df)} valid keypoints (need ≥ 6)")
            return None

        line_results = _fit_three_segment_lines(edge_df[['x', 'y', 'z']].values)
        results = {
            'plant_id': gt_file.stem,
            'source_file': gt_file.name,
            'total_keypoints': len(df),
            'edge_keypoints': len(edge_df),
            'line_fitting': line_results,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        print(f"  RMS error: {line_results['quality_metrics']['overall_rms_error']:.4f} m")
        return results

    except Exception as exc:
        print(f"  Failed: {exc}")
        return None


# ── Main processing logic ─────────────────────────────────────────────────────

def _run(gt_folder: Path, output_dir: Path,
         radius: float = 0.1, extension: float = 0.04) -> None:
    """Process all plant CSVs in *gt_folder* and write ``detection_filter_params.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(gt_folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {gt_folder}")

    print(f"Found {len(csv_files)} ground-truth files in {gt_folder}")
    print(f"Output directory: {output_dir}\n")

    cylinders: Dict[str, Any] = {}
    n_ok = 0

    for gt_file in csv_files:
        result = _process_plant(gt_file)
        if result is None:
            continue

        plant_id = result['plant_id']
        segments = result['line_fitting']['segments']
        cyl_entries = []
        for seg_name in ('bottom', 'center', 'top'):
            seg = segments[seg_name]
            pa, pb = _extend_segment(
                np.array(seg['start_point']),
                np.array(seg['end_point']),
                extension,
            )
            cyl_entries.append({'pointa': pa, 'pointb': pb, 'radius': radius})
        cylinders[plant_id] = cyl_entries
        n_ok += 1
        print()

    params_file = output_dir / 'detection_filter_params.json'
    with open(params_file, 'w') as f:
        json.dump(cylinders, f, indent=2)

    print(f"Processing complete: {n_ok}/{len(csv_files)} plants fitted successfully")
    print(f"Detection filter parameters saved to: {params_file}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="wur-av-bench fit-stem-lines",
        description="Fit piecewise linear segments to plant stem edge keypoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("gt_folder", type=Path,
                   help="Folder containing ground-truth CSV files (one per plant)")
    p.add_argument("--output", dest="output_dir", type=Path,
                   default=Path("fit_stem_lines_output"),
                   help="Output directory for results (default: fit_stem_lines_output)")
    p.add_argument("--radius", type=float, default=0.1,
                   help="Cylinder radius written to the JSON in metres (default: 0.1)")
    p.add_argument("--extension", type=float, default=0.04,
                   help="Length added to each segment end for cylinder endpoints in metres (default: 0.04)")
    args = p.parse_args(argv)

    _run(args.gt_folder, args.output_dir, radius=args.radius, extension=args.extension)


if __name__ == "__main__":
    main()

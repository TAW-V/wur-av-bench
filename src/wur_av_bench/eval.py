#!/usr/bin/env python3
"""
Evaluate 3D/5D world-model reconstructions against ground-truth annotations across
one or more experiments.

For each plant, predicted nodes are matched to ground-truth nodes via cumulative
Hungarian matching: two class-specific passes (petiole, peduncle) followed by an
optional class-agnostic fallback pass.  Precision, recall, and F1 are accumulated
across viewpoints and reported per plant and in aggregate.

A planner that samples viewpoints at random gives a different result every time, so a
plant can be evaluated over several runs.  Name the files `<plant>_run<k>.json` and every
run is evaluated separately; `<plant>.json` is read as a single run.  Results are first
averaged over the runs of a plant and then summarised over plants, and the uncertainty
comes from a hierarchical bootstrap over plants and runs (see stats.py).

Usage
-----
    wur-av-bench eval --gt GT_DIR --output OUT_DIR \\
        --experiments Active:PRED_DIR_A Random:PRED_DIR_B

Arguments
---------
--gt PATH                   Ground-truth directory (one CSV per plant).
                            Columns: ID, x, y, z, qx, qy, qz, qw, class,
                            occlusion_level, plant_complexity_level.
--output PATH               Output directory for figures and TSVs.
--experiments NAME:PATH …   One or more experiment name + prediction-folder pairs.
                            Prediction folder contains `<plant>.json` or `<plant>_run<k>.json`.
--distance_threshold M      Max Euclidean distance (m) to count a match (default: 0.05).
--yaw_threshold DEG         Yaw tolerance when --orientation_check is set (default: 20).
--pitch_threshold DEG       Pitch tolerance when --orientation_check is set (default: 30).
--orientation_check         Gate matching on yaw/pitch in addition to distance.
--class_agnostic_fallback   Enable the class-agnostic third matching pass.
--band KIND                 Band in the curve figures: quantile (simultaneous, from the
                            bootstrap percentiles, default), supt (simultaneous, symmetric),
                            chi2 (simultaneous, classical ellipsoid), pointwise, or iqr
                            (spread between plants, the old behaviour).
--bootstrap N               Bootstrap resamples (default: 2000).
--ci FRACTION               Confidence level (default: 0.95).
--boot_seed N               Seed of the bootstrap, so figures are reproducible (default: 0).
--viz_all_runs              Write 3D visualisations for every run instead of the first one.
--paired_test               Add a Wilcoxon p value to the paired comparison.
--recall_threshold F        Recall for the viewpoint efficiency figure and table (default: 0.30).

Matching
--------
Cumulative two-pass Hungarian per viewpoint:
    Pass 1 - petiole↔petiole, then peduncle↔peduncle (class-specific).
    Pass 2 - class-agnostic fallback for remaining unmatched nodes (optional).
GT nodes matched at viewpoint i cannot be re-matched at viewpoint j > i.
Precision = cumulative_TP / (cumulative_TP + FP_at_current_VP)
Recall    = cumulative_TP / total_GT_nodes
F1        = harmonic mean of precision and recall

Author : Thomas Versmissen <thomas.versmissen@wur.nl>
License: MIT - see LICENSE for full text
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d

from wur_av_bench import stats

# Constants
THIRD_NAMES  = ['t1', 't2', 't3']
THIRD_LABELS = ['T1 (Low)', 'T2 (Mid)', 'T3 (High)']
EXP_COLORS   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
MATCH_COLORS = [(1.,1.,1.), (0.,0.,0.), (1.,.75,.8), (.96,.96,.86), (0.,1.,0.)]

# Check for LaTeX availability for nicer plots; fall back to default math font if not found.
def _latex_available() -> bool:
    """True only when latex is installed and matplotlib can really render with it.

    Finding the binary is not enough. A partial texlive install misses packages that matplotlib
    needs (type1cm for instance), and then the run dies halfway through the first figure. So try
    one small string and fall back to the default fonts when that fails.
    """
    import shutil
    if shutil.which("latex") is None:
        return False
    try:
        from matplotlib.texmanager import TexManager
        TexManager().make_dvi("test", 10)
        return True
    except Exception:
        return False

_use_tex = _latex_available()
print(f"Matplotlib LaTeX rendering {'enabled' if _use_tex else 'disabled'} ")
# LaTeX needs '%' escaped, plain text must not escape it.
_PCT = r'\%' if _use_tex else '%'

plt.rcParams.update({
    "text.usetex": _use_tex, "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    **({} if _use_tex else {"mathtext.fontset": "cm"}),
})


# Data structure
@dataclass
class PlantResult:
    """Arrays of one run of one plant, produced by cumulative viewpoint matching."""
    vp_metrics:  np.ndarray  # (V, 3)  precision / recall / f1 per viewpoint
    m_gi:        np.ndarray  # (M,)    GT row index for each cumulative match
    m_pred_xyz:  np.ndarray  # (M, 3)  matched pred positions
    m_pred_quat: np.ndarray  # (M, 4)  matched pred quaternions
    m_pred_id:   np.ndarray  # (M,)    matched pred node IDs (str)
    m_dist:      np.ndarray  # (M,)    position errors (m)
    u_pred_xyz:  np.ndarray  # (U, 3)  last-vp unmatched pred positions
    u_pred_quat: np.ndarray  # (U, 4)  last-vp unmatched pred quaternions
    u_pred_id:   np.ndarray  # (U,)    last-vp unmatched pred IDs
    g_xyz:       np.ndarray  # (G, 3)
    g_quat:      np.ndarray  # (G, 4)
    g_cls:       np.ndarray  # (G,)    str
    g_orig_id:   np.ndarray  # (G,)    str  original CSV node IDs
    g_global_id: np.ndarray  # (G,)    str  "{plant_id}_{orig_id}"
    g_occ:       np.ndarray  # (G,)    float  occlusion_level per GT node
    run:         int = 0     #         run number of this plant


# ---------------------------------------------------
# Geometry
# ---------------------------------------------------

def orientation_error(q_pred: np.ndarray, q_gt: np.ndarray
                      ) -> Tuple[Optional[float], Optional[float]]:
    """Signed yaw and pitch error (in degrees). Returns (None, None) on invalid quat."""
    try:
        v_p = R.from_quat(q_pred).apply([1., 0., 0.])
        v_g = R.from_quat(q_gt).apply([1., 0., 0.])
    except ValueError:
        return None, None
    yaw   = ((np.degrees(np.arctan2(v_g[0]*v_p[1] - v_g[1]*v_p[0],
                                    v_g[0]*v_p[0] + v_g[1]*v_p[1])) + 180) % 360) - 180
    pitch = ((np.degrees(np.arcsin(np.clip(v_p[2], -1., 1.))
                       - np.arcsin(np.clip(v_g[2], -1., 1.))) + 180) % 360) - 180
    return float(yaw), float(pitch)


# ---------------------------------------------------
# Matching
# ---------------------------------------------------

def _one_pass(
        p_xyz: np.ndarray, g_xyz: np.ndarray,
        p_cls: np.ndarray, g_cls: np.ndarray,
        p_quats: np.ndarray, g_quats: np.ndarray,
        p_avail: np.ndarray, g_avail: np.ndarray,   # bool, modified in-place
        dist_thresh: float, yaw_thresh: float, pitch_thresh: float,
        orientation_check: bool, class_filter: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single Hungarian pass. Returns (p_indices, g_indices, distances).
    Marks matched nodes unavailable in p_avail / g_avail (in-place).

    args:
        p_xyz: (P, 3) predicted node positions.
        g_xyz: (G, 3) GT node positions.
        p_cls: (P,) predicted node class labels.
        g_cls: (G,) GT node class labels.
        p_quats: (P, 4) predicted node orientations as quaternions.
        g_quats: (G, 4) GT node orientations as quaternions.
        p_avail: (P,) bool mask of unmatched predicted nodes; updated in-place.
        g_avail: (G,) bool mask of unmatched GT nodes; updated in-place.
        dist_thresh: maximum Euclidean distance (metres) to accept a match.
        yaw_thresh: maximum absolute yaw error (degrees) when orientation_check is True.
        pitch_thresh: maximum absolute pitch error (degrees) when orientation_check is True.
        orientation_check: if True, discard matches that exceed yaw/pitch thresholds.
        class_filter: if given, only consider nodes of this class; if None, all available nodes are used.

    returns:
        p_indices: indices into p_xyz of matched predictions.
        g_indices: indices into g_xyz of matched GT nodes.
        distances: Euclidean distances for each matched pair.
    """
    pm = p_avail & (p_cls == class_filter) if class_filter else p_avail
    gm = g_avail & (g_cls == class_filter) if class_filter else g_avail
    pi, gi = np.where(pm)[0], np.where(gm)[0]
    if not len(pi) or not len(gi):
        return np.empty(0, int), np.empty(0, int), np.empty(0)

    D  = cdist(p_xyz[pi], g_xyz[gi]) # compute euclidean distance matrix for available preds and GTs of the current class
    D[D > dist_thresh] = 1e6 # gate above-threshold pairs so the assignment maximizes within-threshold matches instead of minimizing total distance
    ri, ci = linear_sum_assignment(D) # Hungarian algorithm to find optimal matching. ri=row indices into pi, ci=col indices into gi, so pi[ri], gi[ci] are the matched pairs in the original p/g arrays.
    dists  = D[ri, ci] # extract distances for matched pairs
    valid  = dists <= dist_thresh # filter matches by distance threshold

    if orientation_check and valid.any():
        keep = [k for k, (r, c) in enumerate(zip(ri[valid], ci[valid]))  # k: position within the distance-valid subset
                if (lambda ye, pe: ye is not None                         
                    and abs(ye) <= yaw_thresh and abs(pe) <= pitch_thresh  
                    )(*orientation_error(p_quats[pi[r]], g_quats[gi[c]]))]
        sel = np.where(valid)[0][keep]  # map back to indices in the full ri/ci arrays
        ri, ci, dists = ri[sel], ci[sel], dists[sel]  # keep only orientation-valid matches
    else:
        ri, ci, dists = ri[valid], ci[valid], dists[valid]  # keep only distance-valid matches

    p_avail[pi[ri]] = False
    g_avail[gi[ci]] = False
    return pi[ri], gi[ci], dists


def match_viewpoint(
        p_xyz: np.ndarray, g_xyz: np.ndarray,
        p_cls: np.ndarray, g_cls: np.ndarray,
        p_quats: np.ndarray, g_quats: np.ndarray,
        p_ids: np.ndarray,
        cum_p_ids: Set[str], g_matched: np.ndarray,  # bool (G,), updated in-place
        dist_thresh: float, yaw_thresh: float, pitch_thresh: float,
        orientation_check: bool, class_agnostic_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Two-pass cumulative matching for one viewpoint. Updates g_matched in-place.
    Pass 1: class-specific (petiole, peduncle).
    Pass 2 (optional): class-agnostic fallback for remaining unmatched nodes.

    args:
        p_xyz: (P, 3) predicted node positions.
        g_xyz: (G, 3) GT node positions.
        p_cls: (P,) predicted node class labels.
        g_cls: (G,) GT node class labels.
        p_quats: (P, 4) predicted node orientations as quaternions.
        g_quats: (G, 4) GT node orientations as quaternions.
        p_ids: (P,) unique string IDs for predicted nodes.
        cum_p_ids: set of prediction IDs already matched in earlier viewpoints.
        g_matched: (G,) bool mask of GT nodes matched so far; updated in-place.
        dist_thresh: maximum Euclidean distance (metres) to accept a match.
        yaw_thresh: maximum absolute yaw error (degrees) when orientation_check is True.
        pitch_thresh: maximum absolute pitch error (degrees) when orientation_check is True.
        orientation_check: if True, discard matches that exceed yaw/pitch thresholds.
        class_agnostic_fallback: if True, run a second pass ignoring class labels for still-unmatched nodes.

    returns:
        p_indices: indices into p_xyz of matched predictions.
        g_indices: indices into g_xyz of matched GT nodes.
        distances: Euclidean distances for each matched pair.
        fp_count: number of predicted nodes that remained unmatched after all passes.
    """
    p_avail = np.array([pid not in cum_p_ids for pid in p_ids])
    g_avail = ~g_matched   # GT nodes not yet claimed by a previous viewpoint; new array so g_matched is not affected

    classes = ('petiole', 'peduncle', None) if class_agnostic_fallback \
              else ('petiole', 'peduncle')
    pis, gis, ds = [], [], []
    for cls in classes:
        p, g, d = _one_pass(p_xyz, g_xyz, p_cls, g_cls, p_quats, g_quats,
                             p_avail, g_avail,
                             dist_thresh, yaw_thresh, pitch_thresh,
                             orientation_check, cls)
        pis.append(p); gis.append(g); ds.append(d)

    fp   = int(p_avail.sum())   # new preds still unmatched after all passes
    pi_m = np.concatenate(pis).astype(int)  # astype prevents against float on all empty concat
    gi_m = np.concatenate(gis).astype(int)
    g_matched[gi_m] = True
    return pi_m, gi_m, np.concatenate(ds), fp


# ---------------------------------------------------
# Data loading
# ---------------------------------------------------

def load_gt(gt_folder: Path) -> Dict[str, pd.DataFrame]:
    gt = {f.stem: pd.read_csv(f) for f in sorted(gt_folder.glob("*.csv"))}
    print(f"Loaded GT for {len(gt)} plants from {gt_folder}")
    return gt


def load_pred_viewpoints(pred_file: Path) -> List[Tuple[str, dict]]:
    with open(pred_file) as f:
        data = json.load(f)
    return [(list(d.keys())[0], list(d.values())[0]) for d in data]


RUN_PATTERN = re.compile(r"^(?P<plant>.+)_run(?P<run>\d+)$")


def find_prediction_files(pred_folder: Path, gt_dict: Dict[str, pd.DataFrame]
                          ) -> Dict[str, List[Tuple[int, Path]]]:
    """Group the prediction files of one experiment per plant.

    `<plant>_run<k>.json` are runs of the same plant, `<plant>.json` counts as run 0.
    Files of plants without ground truth are skipped.
    """
    per_plant: Dict[str, List[Tuple[int, Path]]] = {}
    for f in sorted(pred_folder.glob("*.json")):
        m = RUN_PATTERN.match(f.stem)
        plant, run = (m.group("plant"), int(m.group("run"))) if m else (f.stem, 0)
        if plant not in gt_dict:
            continue
        per_plant.setdefault(plant, []).append((run, f))
    for runs in per_plant.values():
        runs.sort()
    return per_plant


# ---------------------------------------------------
# Tercile computation
# ---------------------------------------------------

def _assign_terciles(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Label each value 't1'/'t2'/'t3' by rank (equal-count split). Returns (labels, q1, q2)."""
    groups = np.array_split(np.argsort(values, kind='stable'), 3)
    labels = np.empty(len(values), dtype=object)
    for name, idx in zip(THIRD_NAMES, groups):
        labels[idx] = name
    q1 = float(values[groups[0][-1]]) if len(groups[0]) else np.nan
    q2 = float(values[groups[1][-1]]) if len(groups[1]) else np.nan
    return labels, q1, q2


def compute_tercile_maps(gt_dict: Dict[str, pd.DataFrame]
                         ) -> Tuple[Dict, Dict, Dict, Dict, float, float]:
    """Returns node_tercile, plant_cmp_tercile, plant_cmp_value, nq1, nq2."""
    no_cmp = sorted(pid for pid, df in gt_dict.items()
                    if 'plant_complexity_level' not in df.columns)
    if no_cmp:
        print(f"Warning: {len(no_cmp)} GT plant(s) missing 'plant_complexity_level'; "
              f"complexity defaults to 0.0, so complexity terciles will be arbitrary: "
              f"{', '.join(no_cmp)}")
        cont = input("Continue anyway? (y/n): ").strip().lower()
        if cont != 'y':
            print("Aborting. Please add the missing column and try again.")
            exit(1)

    gids, pids, occs, cmps = [], [], [], []
    for pid, df in gt_dict.items():
        for _, row in df.iterrows():
            gids.append(f"{pid}_{row['ID']}")
            pids.append(pid)
            occs.append(float(row['occlusion_level']))
            cmps.append(float(row.get('plant_complexity_level', 0.0)))

    gids = np.array(gids); pids = np.array(pids)
    occs = np.array(occs); cmps = np.array(cmps)

    node_t, nq1, nq2 = _assign_terciles(occs)
    node_tercile = dict(zip(gids, node_t))

    unique = np.unique(pids)
    plant_cmp = np.array([cmps[pids == p][0] for p in unique])
    pc, _, _ = _assign_terciles(plant_cmp)
    plant_cmp_tercile = dict(zip(unique, pc))
    plant_cmp_value   = dict(zip(unique, plant_cmp))

    return node_tercile, plant_cmp_tercile, plant_cmp_value, nq1, nq2


# ---------------------------------------------------
# Per-plant processing
# ---------------------------------------------------

def process_plant(plant_id: str, pred_viewpoints: List[Tuple[str, dict]],
                  gt_df: pd.DataFrame,
                  dist_thresh: float, yaw_thresh: float,
                  pitch_thresh: float, orientation_check: bool,
                  class_agnostic_fallback: bool = False) -> PlantResult:
    """Cumulative viewpoint-by-viewpoint matching for a single plant."""
    g_global_id = np.array([f"{plant_id}_{row['ID']}" for _, row in gt_df.iterrows()])
    g_orig_id   = gt_df['ID'].astype(str).values
    g_xyz       = gt_df[['x', 'y', 'z']].values.astype(float)
    g_quat      = gt_df[['qx', 'qy', 'qz', 'qw']].values.astype(float)
    g_cls       = gt_df['class'].astype(str).values
    g_occ       = gt_df['occlusion_level'].values.astype(float)
    n_gt        = len(gt_df)

    g_matched   = np.zeros(n_gt, dtype=bool)
    cum_p_ids:  Set[str] = set()
    vp_rows     = []

    m_gi_l, m_pxyz_l, m_pquat_l, m_pid_l, m_dist_l = [], [], [], [], []
    u_pred_xyz  = np.empty((0, 3))
    u_pred_quat = np.empty((0, 4))
    u_pred_id   = np.empty(0, dtype=object)

    for _, vp_nodes in pred_viewpoints:
        tp = int(g_matched.sum())
        if not vp_nodes:
            prec = 1.0 if tp > 0 else 0.0
            rec  = tp / n_gt if n_gt else 0.0
            vp_rows.append([prec, rec, 2*prec*rec/(prec+rec) if prec+rec else 0.])
            u_pred_xyz  = np.empty((0, 3))
            u_pred_quat = np.empty((0, 4))
            u_pred_id   = np.empty(0, dtype=object)
            continue

        p_ids_l = list(vp_nodes.keys())
        p_ids   = np.array(p_ids_l)
        p_xyz   = np.array([[vp_nodes[i]['x'], vp_nodes[i]['y'], vp_nodes[i]['z']]
                             for i in p_ids_l])
        p_quats = np.array([[vp_nodes[i]['qx'], vp_nodes[i]['qy'],
                              vp_nodes[i]['qz'], vp_nodes[i]['qw']]
                             for i in p_ids_l])
        p_cls   = np.array([vp_nodes[i]['class'] for i in p_ids_l])

        pi_m, gi_m, dist_m, fp = match_viewpoint(
            p_xyz, g_xyz, p_cls, g_cls, p_quats, g_quat, p_ids,
            cum_p_ids, g_matched,
            dist_thresh, yaw_thresh, pitch_thresh, orientation_check,
            class_agnostic_fallback,
        )

        # Unmatched new preds — compute before updating cum_p_ids
        new_mask = np.array([pid not in cum_p_ids for pid in p_ids_l])
        mp_mask  = np.zeros(len(p_ids), dtype=bool)
        if len(pi_m):
            mp_mask[pi_m] = True
        umask       = new_mask & ~mp_mask
        u_pred_xyz  = p_xyz[umask]
        u_pred_quat = p_quats[umask]
        u_pred_id   = p_ids[umask]

        if len(pi_m):
            m_gi_l.append(gi_m);          m_pxyz_l.append(p_xyz[pi_m])
            m_pquat_l.append(p_quats[pi_m]); m_pid_l.append(p_ids[pi_m])
            m_dist_l.append(dist_m)
            cum_p_ids.update(p_ids[pi_m])

        tp   = int(g_matched.sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.
        rec  = tp / n_gt if n_gt else 0.
        vp_rows.append([prec, rec, 2*prec*rec/(prec+rec) if prec+rec else 0.])

    def _cat(lst, empty_shape):
        return np.concatenate(lst) if lst else np.empty(empty_shape)

    return PlantResult(
        vp_metrics  = np.array(vp_rows) if vp_rows else np.empty((0, 3)),
        m_gi        = _cat(m_gi_l,    (0,)).astype(int),
        m_pred_xyz  = _cat(m_pxyz_l,  (0, 3)),
        m_pred_quat = _cat(m_pquat_l, (0, 4)),
        m_pred_id   = _cat(m_pid_l,   (0,)).astype(str),
        m_dist      = _cat(m_dist_l,  (0,)),
        u_pred_xyz  = u_pred_xyz,
        u_pred_quat = u_pred_quat,
        u_pred_id   = u_pred_id,
        g_xyz       = g_xyz,
        g_quat      = g_quat,
        g_cls       = g_cls,
        g_orig_id   = g_orig_id,
        g_global_id = g_global_id,
        g_occ       = g_occ,
    )


def run_experiment(pred_folder: Path, gt_dict: Dict[str, pd.DataFrame],
                   dist_thresh: float, yaw_thresh: float,
                   pitch_thresh: float, orientation_check: bool,
                   class_agnostic_fallback: bool = False,
                   ) -> Dict[str, List[PlantResult]]:
    """Evaluate every run of every plant. Returns the runs per plant, in run order."""
    files_per_plant = find_prediction_files(pred_folder, gt_dict)
    if not files_per_plant:
        raise ValueError(f"No matching prediction JSON files found in {pred_folder}")

    per_plant: Dict[str, List[PlantResult]] = {}
    for pid in sorted(files_per_plant):
        results = []
        for run, pf in files_per_plant[pid]:
            r = process_plant(pid, load_pred_viewpoints(pf), gt_dict[pid],
                              dist_thresh, yaw_thresh, pitch_thresh,
                              orientation_check, class_agnostic_fallback)
            r.run = run
            results.append(r)
        per_plant[pid] = results

    if per_plant:
        n_runs = [len(v) for v in per_plant.values()]
        avg_vp = np.mean([len(r.vp_metrics) for runs in per_plant.values() for r in runs])
        runs_text = (f"{n_runs[0]} runs each" if len(set(n_runs)) == 1
                     else f"{min(n_runs)} to {max(n_runs)} runs")
        print(f"  Processed {len(per_plant)} plants, {runs_text}, avg {avg_vp:.1f} viewpoints/run")
        if len(set(n_runs)) != 1:
            uneven = sorted(p for p, v in per_plant.items() if len(v) != max(n_runs))
            print(f"  Warning: not every plant has the same number of runs: {', '.join(uneven)}")
    else:
        print("  Processed 0 plants")

    missing = sorted(set(gt_dict) - set(per_plant))
    if missing:
        print(f"  Warning: {len(missing)} GT plant(s) have no prediction file "
              f"and are excluded from metrics: {', '.join(missing)}")
        cont = input("Continue with available plants? (y/n): ").strip().lower()
        if cont != 'y':
            print("Aborting. Please ensure prediction files are named correctly and try again.")
            exit(1)

    return per_plant


# ---------------------------------------------------
# TSV output
# ---------------------------------------------------

def save_node_matches_tsv(plant_id: str, r: PlantResult, out_dir: Path,
                          plant_complexity: float, plant_complexity_category: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ('pred_id\tpred_x\tpred_y\tpred_z'
              '\tgt_id\tgt_x\tgt_y\tgt_z'
              '\tdistance_error\tyaw_error_deg\tpitch_error_deg'
              '\tocclusion_level'
              '\tplant_complexity_level\tplant_complexity_category')
    plant_suffix = f"\t{plant_complexity:.6f}\t{plant_complexity_category}"
    rows = []

    for k in range(len(r.m_gi)):
        gi = r.m_gi[k]
        ye, pe = orientation_error(r.m_pred_quat[k], r.g_quat[gi])
        ye_s = f"{ye:.4f}" if ye is not None else ""
        pe_s = f"{pe:.4f}" if pe is not None else ""
        px, py, pz = r.m_pred_xyz[k];  gx, gy, gz = r.g_xyz[gi]
        rows.append(f"{r.m_pred_id[k]}\t{px:.6f}\t{py:.6f}\t{pz:.6f}"
                    f"\t{r.g_orig_id[gi]}\t{gx:.6f}\t{gy:.6f}\t{gz:.6f}"
                    f"\t{r.m_dist[k]:.6f}\t{ye_s}\t{pe_s}"
                    f"\t{r.g_occ[gi]:.6f}{plant_suffix}")

    for k in range(len(r.u_pred_id)):
        x, y, z = r.u_pred_xyz[k]
        rows.append(f"{r.u_pred_id[k]}\t{x:.6f}\t{y:.6f}\t{z:.6f}\t\t\t\t\t\t\t\t{plant_suffix}")

    matched_gi = set(r.m_gi)
    for gi in range(len(r.g_orig_id)):
        if gi not in matched_gi:
            gx, gy, gz = r.g_xyz[gi]
            rows.append(f"\t\t\t\t{r.g_orig_id[gi]}\t{gx:.6f}\t{gy:.6f}\t{gz:.6f}\t\t\t"
                        f"\t{r.g_occ[gi]:.6f}{plant_suffix}")

    with open(out_dir / f"{plant_id}.tsv", 'w') as f:
        f.write(header + '\n' + '\n'.join(rows) + '\n')


def save_metrics_summary_tsv(per_plant: Dict[str, List[PlantResult]], out_path: Path,
                             plant_cmp_value: Dict[str, float],
                             plant_cmp_tercile: Dict[str, str]) -> None:
    """One row per (plant, run) with the metrics at the last viewpoint of that run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('plant_id\trun\tviewpoints\tprecision\trecall\tf1'
                '\tplant_complexity_level\tplant_complexity_category\n')
        for pid in sorted(per_plant):
            for r in per_plant[pid]:
                vm = r.vp_metrics
                if not len(vm):
                    continue
                cmp_val = plant_cmp_value.get(pid, float('nan'))
                cmp_cat = plant_cmp_tercile.get(pid, '')
                f.write(f"{pid}\t{r.run}\t{len(vm)}\t{vm[-1,0]:.6f}\t{vm[-1,1]:.6f}\t{vm[-1,2]:.6f}"
                        f"\t{cmp_val:.6f}\t{cmp_cat}\n")


def save_metrics_per_plant_tsv(per_plant: Dict[str, List[PlantResult]], out_path: Path,
                               plant_cmp_value: Dict[str, float],
                               plant_cmp_tercile: Dict[str, str]) -> None:
    """One row per plant: mean, standard deviation, minimum and maximum over its runs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = ('precision', 'recall', 'f1')
    with open(out_path, 'w') as f:
        header = ['plant_id', 'runs', 'viewpoints_mean']
        for n in names:
            header += [f"{n}_mean", f"{n}_sd", f"{n}_min", f"{n}_max"]
        header += ['plant_complexity_level', 'plant_complexity_category']
        f.write('\t'.join(header) + '\n')
        for pid in sorted(per_plant):
            values, index = final_per_plant(per_plant[pid])
            if not len(values):
                continue
            cells = [pid, str(len(values)), f"{index.mean() + 1:.2f}"]
            for col in range(3):
                v = values[:, col]
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                cells += [f"{v.mean():.6f}", f"{sd:.6f}", f"{v.min():.6f}", f"{v.max():.6f}"]
            cells += [f"{plant_cmp_value.get(pid, float('nan')):.6f}", plant_cmp_tercile.get(pid, '')]
            f.write('\t'.join(cells) + '\n')


# ---------------------------------------------------
# 3D Visualization
# ---------------------------------------------------

def _make_cylinder(pos: np.ndarray, quat: np.ndarray,
                   base_color: tuple, top_color: Optional[tuple] = None,
                   length: float = 0.02, radius: float = 0.002,
                   orientation_check: bool = True) -> o3d.geometry.TriangleMesh:
    def _sphere(bc, tc):
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 3)
        if tc is None:
            sp.paint_uniform_color(bc)
        else:
            verts  = np.asarray(sp.vertices)
            colors = np.tile(bc, (len(verts), 1))
            colors[verts[:, 2] >= radius * 3 * 0.6] = tc
            sp.vertex_colors = o3d.utility.Vector3dVector(colors)
        sp.translate(pos); return sp

    if not orientation_check or np.linalg.norm(quat) < 1e-6:
        return _sphere(base_color, top_color)

    if top_color is None:
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        cyl.paint_uniform_color(base_color)
    else:
        bot = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length * 0.8)
        bot.paint_uniform_color(base_color); bot.translate([0, 0, -length * 0.1])
        top = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length * 0.2)
        top.paint_uniform_color(top_color);  top.translate([0, 0,  length * 0.4])
        cyl = bot + top

    rot = R.from_quat(quat) * R.from_euler('y', -90, degrees=True)
    cyl.rotate(rot.as_matrix(), center=(0, 0, 0)); cyl.translate(pos)
    return cyl


def save_3d_visualization(plant_id: str, r: PlantResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    GT_COLOR, PRED_COLOR = (1., 0., 0.), (0., 0., 1.)
    mesh = o3d.geometry.TriangleMesh()
    matched_gi = set(r.m_gi)

    for k, gi in enumerate(r.m_gi):
        color = MATCH_COLORS[k % len(MATCH_COLORS)]
        mesh += _make_cylinder(r.g_xyz[gi],     r.g_quat[gi],     GT_COLOR,   color)
        mesh += _make_cylinder(r.m_pred_xyz[k], r.m_pred_quat[k], PRED_COLOR, color)

    for gi in range(len(r.g_orig_id)):
        if gi not in matched_gi:
            mesh += _make_cylinder(r.g_xyz[gi], r.g_quat[gi], GT_COLOR)

    for k in range(len(r.u_pred_id)):
        mesh += _make_cylinder(r.u_pred_xyz[k], r.u_pred_quat[k], PRED_COLOR)

    if len(mesh.triangles) == 0:
        print(f"    Warning: empty mesh for {plant_id}, skipping"); return
    o3d.io.write_triangle_mesh(str(out_dir / f"{plant_id}_comparison.obj"), mesh)
    o3d.io.write_triangle_mesh(str(out_dir / f"{plant_id}_comparison.ply"), mesh)

# ---------------------------------------------------
# Plot helpers
# ---------------------------------------------------

@dataclass
class BandConfig:
    """How the uncertainty in the figures and tables is computed."""
    kind: str = "quantile"  # quantile, supt, chi2, pointwise or iqr (spread between plants)
    n_boot: int = 2000
    ci: float = 0.95
    seed: int = 0

    def label(self) -> str:
        if self.kind == "iqr":
            return "25-75% over plants"
        name = {"quantile": "simultaneous", "supt": "simultaneous, symmetric",
                "chi2": "simultaneous, chi2", "pointwise": "pointwise"}[self.kind]
        return f"{int(self.ci * 100)}% {name} band"


def max_viewpoints(all_exp: Dict[str, Dict[str, List[PlantResult]]]) -> int:
    """Longest run anywhere, so all curves can be compared on the same axis."""
    lengths = [len(r.vp_metrics) for pp in all_exp.values() for runs in pp.values() for r in runs]
    return max(lengths) if lengths else 0


def run_curves(runs: List[PlantResult], length: int) -> np.ndarray:
    """(runs, length, 3) metrics of one plant, runs that stopped early carry their last value."""
    out = np.full((len(runs), length, 3), np.nan)
    for i, r in enumerate(runs):
        for col in range(3):
            out[i, :, col] = stats.pad_curves([r.vp_metrics[:, col]], length)[0]
    return out


def experiment_curves(per_plant: Dict[str, List[PlantResult]], length: int,
                      plants: Optional[Set[str]] = None) -> List[np.ndarray]:
    """One (runs, length, 3) array per plant, for the plants asked for."""
    return [run_curves(runs, length) for pid, runs in per_plant.items()
            if len(runs) and len(runs[0].vp_metrics) and (plants is None or pid in plants)]


def final_per_plant(runs: List[PlantResult]) -> Tuple[np.ndarray, np.ndarray]:
    """Per run the final metrics (runs, 3) and the index of the last viewpoint (runs,)."""
    values = np.array([r.vp_metrics[-1] for r in runs if len(r.vp_metrics)])
    index = np.array([len(r.vp_metrics) - 1 for r in runs if len(r.vp_metrics)])
    return values, index


def curve_summary(curves: List[np.ndarray], cfg: BandConfig):
    """Median over plants of the per plant run mean, with a band.

    Returns (median (V, 3), low (V, 3), high (V, 3), factor). With kind 'iqr' the band is the
    25-75% range over plants, as in the single run version of this tool. Precision, recall and F1
    lie between 0 and 1, so the band is cut off there.
    """
    if not curves:
        return None
    per_plant = np.array([np.nanmean(c, axis=0) for c in curves])    # (plants, V, 3)
    median = np.median(per_plant, axis=0)
    if cfg.kind == "iqr":
        return median, np.percentile(per_plant, 25, axis=0), np.percentile(per_plant, 75, axis=0), None

    _, replicates = stats.bootstrap(curves, lambda x: np.median(x, axis=0), cfg.n_boot, cfg.seed)
    low = np.empty_like(median)
    high = np.empty_like(median)
    factor = None
    for col in range(median.shape[1]):
        lo, hi, factor = stats.band(median[:, col], replicates[:, :, col], cfg.kind, cfg.ci)
        low[:, col], high[:, col] = lo, hi
    return median, np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0), factor


def _plot_band(ax, vp, med, low, high, col, label, color):
    """col: 0=Precision, 1=Recall, 2=F1."""
    ax.plot(vp, med[:, col], label=label, color=color)
    ax.fill_between(vp, low[:, col], high[:, col], alpha=0.2, color=color)


def _add_final_dots(ax, per_plant: Dict[str, List[PlantResult]], col: int, color):
    """One dot per plant, where its runs ended on average, with 1 sd bars over the runs.

    Runs of a plant differ in their final value but also in how many viewpoints they used, because
    the planner can reach the top early. Both get a bar, so the horizontal one is the spread in run
    length and the vertical one the spread in the metric.
    """
    for runs in per_plant.values():
        values, index = final_per_plant(runs)
        if not len(values):
            continue
        many = len(values) > 1
        ax.errorbar(index.mean(), values[:, col].mean(),
                    yerr=values[:, col].std(ddof=1) if many else None,
                    xerr=index.std(ddof=1) if many else None,
                    fmt='o', ms=4, color=color, alpha=0.7, elinewidth=1, capsize=2, zorder=5)


def _grouped_boxplot(ax, data_list, positions, colors, tick_positions, tick_labels,
                     ylabel='', title='', fontsize=12):
    bp = ax.boxplot(data_list, positions=positions, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray',
                                    markeredgecolor='black', markersize=3))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize - 1)
    ax.grid(True, axis='y', alpha=0.3); ax.set_facecolor('white')


def _exp_color(idx): return EXP_COLORS[idx % len(EXP_COLORS)]
def _legend_patches(names):
    return [plt.Rectangle((0,0), 1, 1, facecolor=EXP_COLORS[i % len(EXP_COLORS)],
                           alpha=0.7, edgecolor='black') for i in range(len(names))]


# ---------------------------------------------------
# Figures
# ---------------------------------------------------

def fig_avg_prec_rec_f1(all_exp: Dict[str, Dict[str, List[PlantResult]]], out_dir: Path,
                        cfg: BandConfig) -> None:
    """avg_precision_recall_f1_over_vp.pdf — median over plants with a band, 1×3."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    length = max_viewpoints(all_exp)
    vp = np.arange(length)
    for ei, (exp_name, per_plant) in enumerate(all_exp.items()):
        color = _exp_color(ei)
        summary = curve_summary(experiment_curves(per_plant, length), cfg)
        if summary is None: continue
        med, low, high, _ = summary
        for col, ax in enumerate(axes):
            _plot_band(ax, vp, med, low, high, col, exp_name, color)
            _add_final_dots(ax, per_plant, col, color)

    for ax, title, ylabel in zip(axes,
            ['Median Precision', 'Median Recall', 'Median F1'],
            ['Precision', 'Recall', 'F1 Score']):
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Viewpoint index")
        ax.set_ylim(-0.05, 1.05); ax.grid(True); ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "avg_precision_recall_f1_over_vp.pdf", bbox_inches='tight')
    plt.close(fig)


def fig_avg_by_complexity_third(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                                 plant_cmp_tercile: Dict[str, str], out_dir: Path,
                                 cfg: BandConfig) -> None:
    """avg_precision_recall_f1_over_vp_by_complexity.pdf — 3×3 grid."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey='row')
    length = max_viewpoints(all_exp)
    vp = np.arange(length)

    for col, third in enumerate(THIRD_NAMES):
        for ei, (exp_name, per_plant) in enumerate(all_exp.items()):
            color = _exp_color(ei)
            plants = {pid for pid in per_plant if plant_cmp_tercile.get(pid) == third}
            summary = curve_summary(experiment_curves(per_plant, length, plants), cfg)
            if summary is None: continue
            med, low, high, _ = summary
            third_pp = {pid: runs for pid, runs in per_plant.items() if pid in plants}
            for row in range(3):
                _plot_band(axes[row, col], vp, med, low, high, row, exp_name, color)
                _add_final_dots(axes[row, col], third_pp, row, color)

        for row in range(3):
            axes[row, col].grid(True); axes[row, col].set_ylim(-0.05, 1.05)

    for row, label in enumerate(['Precision', 'Recall', 'F1 Score']):
        axes[row, 0].set_ylabel(label, fontsize=17)
    for col, label in enumerate(THIRD_LABELS):
        axes[0, col].set_title(label, fontsize=17)
    for col in range(3):
        axes[2, col].set_xlabel("Viewpoint index", fontsize=16)
    for ax in axes.flat:
        ax.tick_params(labelsize=14)

    exp_names = list(all_exp.keys())
    fig.legend(_legend_patches(exp_names), exp_names,
               loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(exp_names),
               fontsize=14)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.10)
    fig.savefig(out_dir / "avg_precision_recall_f1_over_vp_by_complexity.pdf", bbox_inches='tight')
    plt.close(fig)


def fig_individual_by_complexity_third(exp_name: str,
                                        per_plant: Dict[str, List[PlantResult]],
                                        plant_cmp_tercile: Dict[str, str],
                                        out_dir: Path) -> None:
    """precision_recall_f1_individual_plants_by_complexity_{exp}.pdf

    One line per plant (mean over its runs) with a light band from the lowest to the
    highest run, so the spread of a single plant is visible.

    Runs of the same plant do not all use the same number of viewpoints, because the planner can
    reach the top of the plant early. Past the shortest run the line and the band therefore carry
    the last value of the runs that already stopped, the same carry forward used elsewhere. The red
    dots mark where each run really ended, so that spread stays visible instead of being hidden
    behind the longest run.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey='row')
    for col, third in enumerate(THIRD_NAMES):
        for pid, runs in per_plant.items():
            if plant_cmp_tercile.get(pid) != third or not len(runs[0].vp_metrics): continue
            length = max(len(r.vp_metrics) for r in runs)
            curves = run_curves(runs, length)          # (runs, length, 3)
            vp_idx = np.arange(length)
            for row in range(3):
                mean = curves[:, :, row].mean(axis=0)
                line, = axes[row, col].plot(vp_idx, mean, alpha=0.6)
                if len(runs) > 1:
                    axes[row, col].fill_between(vp_idx, curves[:, :, row].min(axis=0),
                                                curves[:, :, row].max(axis=0),
                                                color=line.get_color(), alpha=0.12)
                ends = [len(r.vp_metrics) - 1 for r in runs]
                axes[row, col].scatter(ends, [curves[i, e, row] for i, e in enumerate(ends)],
                                       c='red', s=14, alpha=0.7, zorder=5)
        for row in range(3):
            axes[row, col].grid(True); axes[row, col].set_ylim(-0.05, 1.05)
    for row, label in enumerate(['Precision', 'Recall', 'F1 Score']):
        axes[row, 0].set_ylabel(label)
    for col in range(3):
        axes[2, col].set_xlabel("Viewpoint index")
    fig.tight_layout()
    safe  = exp_name.replace(' ', '_').replace('/', '_')
    fname = f"precision_recall_f1_individual_plants_by_complexity_{safe}.pdf"
    fig.savefig(out_dir / fname, bbox_inches='tight')
    plt.close(fig)


def node_hits(runs: List[PlantResult], node_tercile: Dict[str, str],
              third: Optional[str] = None, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """(runs, nodes) with 1 where a GT node of this plant was matched in that run."""
    gids = runs[0].g_global_id
    if mask is None:
        mask = np.array([node_tercile.get(g) == third for g in gids])
    if not mask.any():
        return None
    out = np.zeros((len(runs), int(mask.sum())))
    for i, r in enumerate(runs):
        matched = np.zeros(len(gids), dtype=bool)
        matched[r.m_gi] = True
        out[i] = matched[mask]
    return out


def detection_rate_ci(values: List[np.ndarray], cfg: BandConfig) -> Tuple[float, float, float]:
    """Mean detection rate over nodes (nodes weigh equally) with a hierarchical bootstrap CI.

    values: per plant a (runs, nodes) array of hits. A node found in 12 of 20 runs counts as 0.6,
    which is the recall one run is expected to reach.
    """
    if not values:
        return float('nan'), float('nan'), float('nan')
    estimate = float(np.concatenate([v.mean(axis=0) for v in values]).mean())
    rng = np.random.default_rng(cfg.seed)
    boot = np.empty(cfg.n_boot)
    for b in range(cfg.n_boot):
        pick = rng.integers(0, len(values), len(values))
        parts = [values[p][rng.integers(0, len(values[p]), len(values[p]))].mean(axis=0) for p in pick]
        boot[b] = np.concatenate(parts).mean()
    low, high = stats.pointwise_interval(boot, cfg.ci)
    return estimate, float(low), float(high)


def fig_node_recall_pct_by_third(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                                  node_tercile: Dict[str, str], out_dir: Path,
                                  cfg: BandConfig) -> None:
    """node_recall_percentage_by_occlusion_level.pdf

    Bar = mean detection rate of the nodes in that occlusion tercile, so with several runs a node
    counts as the fraction of runs in which it was found. Error bar = bootstrap interval.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    n_exp, bar_w, x = len(all_exp), 0.8 / len(all_exp), np.arange(3)

    for ei, (exp_name, per_plant) in enumerate(all_exp.items()):
        color = _exp_color(ei)
        pcts, errs = [], [[], []]
        for third in THIRD_NAMES:
            hits = [h for runs in per_plant.values()
                    if (h := node_hits(runs, node_tercile, third)) is not None]
            est, low, high = detection_rate_ci(hits, cfg)
            pcts.append(100 * est)
            errs[0].append(100 * (est - low))
            errs[1].append(100 * (high - est))
        bars = ax.bar(x + ei * bar_w - (n_exp-1)*bar_w/2, pcts, bar_w,
                      yerr=np.array(errs) if cfg.kind != "iqr" else None, capsize=3,
                      label=exp_name, color=color, alpha=0.8, edgecolor='black',
                      error_kw=dict(ecolor='black', lw=1))
        for bar, pct, up in zip(bars, pcts, errs[1]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + up + 1,
                    f'{pct:.1f}{_PCT}', ha='center', va='bottom', fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(THIRD_LABELS)
    ax.set_xlabel('Node Occlusion Level', fontweight='bold')
    ax.set_ylabel(f'Node Recall ({_PCT})', fontweight='bold')
    ax.set_ylim(0, 110); ax.grid(True, axis='y', alpha=0.3); ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_dir / "node_recall_percentage_by_occlusion_level.pdf", bbox_inches='tight')
    plt.close(fig)


def fig_node_error_boxplots(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                             node_tercile: Dict[str, str], out_dir: Path) -> None:
    """node_error_boxplots.png"""
    n_exp = len(all_exp)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (mkey, title, ylabel) in zip(axes, [
            ('pos', 'Position Error (m)', 'Error (m)'),
            ('yaw', 'Yaw Error (°)',      'Error (°)'),
            ('pit', 'Pitch Error (°)',    'Error (°)')]):
        data_list, positions, colors = [], [], []
        tick_positions, tick_labels  = [], []
        for t_idx, third in enumerate(THIRD_NAMES):
            base = t_idx * (n_exp + 1) + 1
            tick_positions.append(base + (n_exp-1)/2); tick_labels.append(THIRD_LABELS[t_idx])
            for ei, (_, per_plant) in enumerate(all_exp.items()):
                vals = []
                for runs in per_plant.values():
                    for r in runs:      # matched nodes of every run
                        for k, gi in enumerate(r.m_gi):
                            if node_tercile.get(r.g_global_id[gi]) != third: continue
                            if mkey == 'pos':
                                vals.append(r.m_dist[k])
                            else:
                                ye, pe = orientation_error(r.m_pred_quat[k], r.g_quat[gi])
                                if mkey == 'yaw' and ye is not None: vals.append(ye)
                                elif mkey == 'pit' and pe is not None: vals.append(pe)
                data_list.append(vals if vals else [np.nan])
                positions.append(base + ei); colors.append(_exp_color(ei))
        _grouped_boxplot(ax, data_list, positions, colors,
                         tick_positions, tick_labels, ylabel=ylabel, title=title)

    exp_names = list(all_exp.keys())
    fig.legend(_legend_patches(exp_names), exp_names,
               loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=n_exp)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_dir / "node_error_boxplots.pdf", bbox_inches='tight')
    plt.close(fig)


def fig_node_recall_vs_occlusion_binned(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                                         gt_dict: Dict[str, pd.DataFrame],
                                         out_dir: Path, bin_size: float = 0.1) -> None:
    """node_recall_vs_occlusion_level_binned.png"""
    node_vis = {f"{pid}_{row['ID']}": float(row['occlusion_level'])
                for pid, df in gt_dict.items() for _, row in df.iterrows()}
    gid_arr = np.array(list(node_vis.keys()))
    vis_arr = np.array(list(node_vis.values()))
    edges   = np.arange(np.floor(vis_arr.min()*10)/10,
                        np.ceil(vis_arr.max()*10)/10 + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=(6, 4))
    for ei, (exp_name, per_plant) in enumerate(all_exp.items()):
        color = _exp_color(ei)
        rate = {}   # per GT node the fraction of runs in which it was found
        for runs in per_plant.values():
            hits = node_hits(runs, {}, mask=np.ones(len(runs[0].g_global_id), dtype=bool))
            rate.update(dict(zip(runs[0].g_global_id, hits.mean(axis=0))))
        det_arr = np.array([rate.get(gid, 0.0) for gid in gid_arr], dtype=float)

        centers, rates, counts = [], [], []
        for i in range(len(edges) - 1):
            mask = (vis_arr >= edges[i]) & (vis_arr < edges[i+1])
            if mask.sum() > 0:
                centers.append((edges[i] + edges[i+1]) / 2)
                rates.append(det_arr[mask].mean() * 100)
                counts.append(int(mask.sum()))
        if centers:
            ax.plot(centers, rates, color=color, linewidth=2,
                    marker='o', markersize=5, label=exp_name, alpha=0.8)
            max_c = max(counts)
            ax.scatter(centers, rates, s=[50 + (c/max_c)*450 for c in counts],
                       color=color, alpha=0.2, zorder=2)

    ax.set_xlabel('Node Occlusion Level', fontweight='bold')
    ax.set_ylabel(f'Node Recall ({_PCT})', fontweight='bold')
    ax.set_ylim(-5, 105); ax.grid(True, alpha=0.3); ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_dir / "node_recall_vs_occlusion_level_binned.pdf", bbox_inches='tight')
    plt.close(fig)


def viewpoints_to_threshold(per_plant: Dict[str, List[PlantResult]], plants: Set[str],
                            threshold: float = 0.30) -> Tuple[List[float], List[np.ndarray]]:
    """When the runs of each plant first reach the recall threshold.

    Returns per plant the median viewpoint over the runs that reached it (only plants where
    more than half of the runs got there), and per plant a (runs, 1) array that is 1 for a
    run that reached it.
    """
    vp_per_plant, runs_reached = [], []
    for pid, runs in per_plant.items():
        if pid not in plants: continue
        reached, flags = [], []
        for r in runs:
            if not len(r.vp_metrics): continue
            at = next((vi+1 for vi, rec in enumerate(r.vp_metrics[:, 1])
                       if rec >= threshold), None)
            flags.append(1.0 if at is not None else 0.0)
            if at is not None:
                reached.append(at)
        if flags:
            runs_reached.append(np.array(flags).reshape(-1, 1))
        if len(reached) * 2 > len(flags):
            vp_per_plant.append(float(np.median(reached)))
    return vp_per_plant, runs_reached


def fig_viewpoints_to_30pct_by_complexity(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                                           plant_cmp_tercile: Dict[str, str],
                                           out_dir: Path, cfg: BandConfig,
                                           threshold: float = 0.30) -> None:
    """viewpoints_to_{pct}pct_by_complexity_boxplot.pdf, pct from the threshold (30 by default)

    Top: share of the runs that reach the threshold.
    Bottom: per plant the median number of viewpoints over the runs that reached it.
    """
    n_exp, exp_names = len(all_exp), list(all_exp.keys())
    fig, (ax_bar, ax_box) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True,
                                          gridspec_kw={'height_ratios': [1, 3]})
    all_data, positions, colors         = [], [], []
    reached_pcts, not_reached_pcts      = [], []
    tick_positions, tick_labels         = [], []

    for t_idx, third in enumerate(THIRD_NAMES):
        base = t_idx * (n_exp + 1) + 1
        tick_positions.append(base + (n_exp-1)/2); tick_labels.append(THIRD_LABELS[t_idx])
        for ei, (_, per_plant) in enumerate(all_exp.items()):
            positions.append(base + ei); colors.append(_exp_color(ei))
            # per plant the median over the runs that reached the threshold; a plant counts as
            # "not reached" when fewer than half of its runs get there
            plants = {pid for pid in per_plant if plant_cmp_tercile.get(pid) == third}
            vp_per_plant, runs_reached = viewpoints_to_threshold(per_plant, plants, threshold)
            all_data.append(vp_per_plant if vp_per_plant else [np.nan])
            est, _, _ = detection_rate_ci(runs_reached, cfg)
            r_pct = 100. * est if runs_reached else 0.
            reached_pcts.append(r_pct); not_reached_pcts.append(100. - r_pct)

    for pos, rp, nrp, color in zip(positions, reached_pcts, not_reached_pcts, colors):
        ax_bar.bar(pos, rp,  0.6, color=color, alpha=0.8, edgecolor='black')
        ax_bar.bar(pos, nrp, 0.6, bottom=rp, color='white', edgecolor='black', hatch='//')
    several_runs = any(len(runs) > 1 for pp in all_exp.values() for runs in pp.values())
    ax_bar.set_ylim(0, 105)
    ax_bar.set_ylabel(f"{'Runs' if several_runs else 'Plants'} ({_PCT})", fontsize=17)
    ax_bar.tick_params(labelsize=14)
    ax_bar.grid(True, axis='y', alpha=0.3); ax_bar.set_facecolor('white')

    _grouped_boxplot(ax_box, all_data, positions, colors, tick_positions, tick_labels,
                     ylabel=f'Viewpoints to reach {round(threshold * 100)}{_PCT} recall',
                     fontsize=17)
    # no run can need more viewpoints than the longest run has
    ax_box.set_ylim(0, max_viewpoints(all_exp) + 0.5)

    not_reached_patch = plt.Rectangle((0,0), 1, 1, facecolor='white',
                                       edgecolor='black', hatch='//')
    fig.legend(_legend_patches(exp_names) + [not_reached_patch],
               exp_names + [f'Did not reach {round(threshold * 100)}{_PCT}'],
               loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=n_exp+1,
               fontsize=14)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.12)
    fig.savefig(out_dir / f"viewpoints_to_{round(threshold * 100)}pct_by_complexity_boxplot.pdf",
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------
# Results table
# ---------------------------------------------------

def save_summary_table_tsv(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                           plant_cmp_tercile: Dict[str, str], out_path: Path,
                           cfg: BandConfig) -> None:
    """All table cells in one file: median over plants with interval, per metric and tercile.

    This is the table of the paper (precision, recall, F1 for overall and the three complexity
    terciles), so the numbers can be copied straight into the manuscript.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scopes = [('overall', None)] + list(zip(THIRD_NAMES, THIRD_NAMES))
    with open(out_path, 'w') as f:
        f.write('experiment\tmetric\tscope\tplants\tmedian\tci_low\tci_high\tmean_run_sd\n')
        for exp_name, per_plant in all_exp.items():
            for col, metric in enumerate(('precision', 'recall', 'f1')):
                for scope, tercile in scopes:
                    values = []
                    for pid, runs in per_plant.items():
                        if tercile is not None and plant_cmp_tercile.get(pid) != tercile:
                            continue
                        final, _ = final_per_plant(runs)
                        if len(final):
                            values.append(final[:, col])
                    if not values:
                        continue
                    est, low, high = stats.median_ci(values, cfg.n_boot, cfg.seed, cfg.ci)
                    f.write(f"{exp_name}\t{metric}\t{scope}\t{len(values)}\t{est:.4f}"
                            f"\t{low:.4f}\t{high:.4f}\t{stats.mean_run_sd(values):.4f}\n")


def save_paired_comparison_tsv(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                               plant_cmp_tercile: Dict[str, str], out_path: Path,
                               cfg: BandConfig, with_test: bool = False) -> None:
    """Compare every pair of experiments on the same plants.

    Two methods are run on the same plants, so they can be compared per plant. The plant to plant
    variation then cancels, which makes the comparison much more sensitive than looking at two
    separate medians. With one experiment there is nothing to compare and no file is written.
    """
    names = list(all_exp)
    if len(names) < 2:
        return
    scopes = [('overall', None)] + list(zip(THIRD_NAMES, THIRD_NAMES))
    rows = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            shared = [p for p in all_exp[name_a] if p in all_exp[name_b]]
            for col, metric in enumerate(('precision', 'recall', 'f1')):
                for scope, tercile in scopes:
                    plants = [p for p in shared
                              if tercile is None or plant_cmp_tercile.get(p) == tercile]
                    values_a, values_b = [], []
                    for p in plants:
                        fa, _ = final_per_plant(all_exp[name_a][p])
                        fb, _ = final_per_plant(all_exp[name_b][p])
                        if len(fa) and len(fb):
                            values_a.append(fa[:, col])
                            values_b.append(fb[:, col])
                    if not values_a:
                        continue
                    est, low, high, a_better, b_better = stats.paired_difference(
                        values_a, values_b, cfg.n_boot, cfg.seed, cfg.ci)
                    row = [f"{name_a} - {name_b}", metric, scope, str(len(values_a)),
                           f"{est:.4f}", f"{low:.4f}", f"{high:.4f}", str(a_better), str(b_better)]
                    if with_test:
                        from scipy.stats import wilcoxon
                        diffs = [np.mean(a) - np.mean(b) for a, b in zip(values_a, values_b)]
                        p_value = wilcoxon(diffs).pvalue if len(set(diffs)) > 1 else float('nan')
                        row.append(f"{p_value:.4g}")
                    rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ['pair', 'metric', 'scope', 'plants', 'median_difference', 'ci_low', 'ci_high',
              'plants_first_better', 'plants_second_better']
    if with_test:
        header.append('wilcoxon_p')
    with open(out_path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for row in rows:
            f.write('\t'.join(row) + '\n')


def save_viewpoints_to_threshold_tsv(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                                     plant_cmp_tercile: Dict[str, str], out_path: Path,
                                     cfg: BandConfig, threshold: float = 0.30) -> None:
    """The numbers behind the viewpoints to threshold figure, per experiment and complexity tercile.

    runs_reached_pct: share of the runs (over all plants) that reach the threshold, with a
    bootstrap interval. plants_in_boxplot: plants where more than half of the runs reach it;
    the median and quartiles are over their per plant medians, as in the boxplot.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scopes = [('overall', None)] + list(zip(THIRD_NAMES, THIRD_NAMES))
    with open(out_path, 'w') as f:
        f.write('experiment\tscope\tplants\truns\truns_reached_pct\tci_low\tci_high'
                '\tplants_in_boxplot\tmedian_viewpoints\tq25_viewpoints\tq75_viewpoints\n')
        for exp_name, per_plant in all_exp.items():
            for scope, tercile in scopes:
                plants = {pid for pid in per_plant
                          if tercile is None or plant_cmp_tercile.get(pid) == tercile}
                vp_per_plant, runs_reached = viewpoints_to_threshold(per_plant, plants, threshold)
                if not runs_reached:
                    continue
                est, low, high = detection_rate_ci(runs_reached, cfg)
                n_runs = sum(len(e) for e in runs_reached)
                if vp_per_plant:
                    q25, med, q75 = np.percentile(vp_per_plant, [25, 50, 75])
                else:
                    q25 = med = q75 = float('nan')
                f.write(f"{exp_name}\t{scope}\t{len(runs_reached)}\t{n_runs}"
                        f"\t{100 * est:.1f}\t{100 * low:.1f}\t{100 * high:.1f}"
                        f"\t{len(vp_per_plant)}\t{med:.2f}\t{q25:.2f}\t{q75:.2f}\n")


def save_curves_tsv(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                    plant_cmp_tercile: Dict[str, str], out_path: Path, cfg: BandConfig) -> None:
    """The curves of the precision, recall and F1 figures, overall and per complexity tercile.

    One row per experiment, scope and number of viewpoints used (1 = after the first
    viewpoint): the median over plants of the per plant run mean, with the same band as in the
    figures (low and high columns).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    length = max_viewpoints(all_exp)
    scopes = [('overall', None)] + list(zip(THIRD_NAMES, THIRD_NAMES))
    with open(out_path, 'w') as f:
        f.write('experiment\tscope\tplants\tviewpoints\tprecision\tprecision_low\tprecision_high'
                '\trecall\trecall_low\trecall_high\tf1\tf1_low\tf1_high\n')
        for exp_name, per_plant in all_exp.items():
            for scope, tercile in scopes:
                plants = {pid for pid in per_plant
                          if tercile is None or plant_cmp_tercile.get(pid) == tercile}
                curves = experiment_curves(per_plant, length, plants)
                summary = curve_summary(curves, cfg)
                if summary is None:
                    continue
                med, low, high, _ = summary
                for v in range(length):
                    cells = [f"{med[v, c]:.4f}\t{low[v, c]:.4f}\t{high[v, c]:.4f}" for c in range(3)]
                    f.write(f"{exp_name}\t{scope}\t{len(curves)}\t{v + 1}\t" + "\t".join(cells) + "\n")


def print_paired_comparison(all_exp: Dict[str, Dict[str, List[PlantResult]]],
                            cfg: BandConfig) -> None:
    """Short version on screen: recall and F1 over all plants, for every pair."""
    names = list(all_exp)
    if len(names) < 2:
        return
    print("\n  Paired comparison on the same plants  (difference of the per plant means)")
    print(f"  {'-' * 78}")
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            shared = [p for p in all_exp[name_a] if p in all_exp[name_b]]
            cells = []
            for col, metric in enumerate(('recall', 'f1'), start=1):
                values_a, values_b = [], []
                for p in shared:
                    fa, _ = final_per_plant(all_exp[name_a][p])
                    fb, _ = final_per_plant(all_exp[name_b][p])
                    if len(fa) and len(fb):
                        values_a.append(fa[:, col])
                        values_b.append(fb[:, col])
                if not values_a:
                    continue
                est, low, high, a_better, b_better = stats.paired_difference(
                    values_a, values_b, cfg.n_boot, cfg.seed, cfg.ci)
                cells.append(f"{metric} {est:+.3f} [{low:+.3f},{high:+.3f}], "
                             f"{name_a} better on {a_better}, {name_b} on {b_better} "
                             f"of {len(values_a)} plants")
            print(f"  {name_a} - {name_b}")
            for c in cells:
                print(f"      {c}")


def print_results_table(
        all_exp: Dict[str, Dict[str, List[PlantResult]]],
        node_tercile: Dict[str, str],
        plant_cmp_tercile: Dict[str, str],
        cfg: BandConfig,
) -> None:
    """Print a compact summary table: metrics as rows, experiments as columns.

    Every number is the median over plants of the per plant mean over runs, with a
    hierarchical bootstrap interval over plants and runs.
    """
    exp_names = list(all_exp.keys())
    col_w = max(22, max(len(n) + 2 for n in exp_names))
    lbl_w = 34
    sep_w = lbl_w + col_w * len(exp_names)

    def hdr(title: str) -> None:
        print(f"\n  {title}")
        print(f"  {'-' * sep_w}")

    def row(label: str, values, fmt=lambda v: f"{v:.3f}") -> None:
        cells = "".join(f"{fmt(v):>{col_w}}" for v in values)
        print(f"  {label:<{lbl_w}}{cells}")

    def final_values_per_plant(per_plant, col, plants=None):
        """Per plant an array with the final metric of every run."""
        return [values[:, col] for pid, runs in per_plant.items()
                if (plants is None or pid in plants) and len((values := final_per_plant(runs)[0]))]

    print()
    print(f"  {'Metric':<{lbl_w}}" + "".join(f"{n:>{col_w}}" for n in exp_names))
    print(f"  {'=' * sep_w}")

    # --- Counts ---
    hdr("Counts")
    row("Plants", [len(pp) for pp in all_exp.values()], fmt=lambda v: str(int(v)))
    row("Runs per plant",
        [sorted({len(runs) for runs in pp.values()}) for pp in all_exp.values()],
        fmt=lambda v: str(v[0]) if len(v) == 1 else f"{min(v)}-{max(v)}")
    row("GT nodes",
        [sum(len(runs[0].g_xyz) for runs in pp.values()) for pp in all_exp.values()],
        fmt=lambda v: str(int(v)))
    row("Matched nodes per run",
        [np.mean([len(r.m_gi) for runs in pp.values() for r in runs]) * len(pp)
         for pp in all_exp.values()],
        fmt=lambda v: f"{v:.1f}")

    # --- Final P / R / F1 ---
    hdr(f"Final metrics at last viewpoint  (median over plants, {int(cfg.ci * 100)}% interval)")
    for ci_col, label in enumerate(["Precision", "Recall", "F1"]):
        cells = []
        for pp in all_exp.values():
            values = final_values_per_plant(pp, ci_col)
            if not values:
                cells.append("nan")
                continue
            est, low, high = stats.median_ci(values, cfg.n_boot, cfg.seed, cfg.ci)
            cells.append(f"{est:.3f} [{low:.2f},{high:.2f}]")
        print(f"  {label:<{lbl_w}}" + "".join(f"{c:>{col_w}}" for c in cells))

    hdr("Variation between runs of one plant  (mean sd)")
    for ci_col, label in enumerate(["Precision", "Recall", "F1"]):
        row(label, [stats.mean_run_sd(final_values_per_plant(pp, ci_col)) for pp in all_exp.values()])

    # --- F1 by complexity tercile ---
    hdr("Final F1 by plant complexity tercile  (median over plants)")
    for t, tlbl in zip(THIRD_NAMES, THIRD_LABELS):
        cells = []
        for pp in all_exp.values():
            plants = {pid for pid in pp if plant_cmp_tercile.get(pid) == t}
            values = final_values_per_plant(pp, 2, plants)
            if not values:
                cells.append("nan")
                continue
            est, low, high = stats.median_ci(values, cfg.n_boot, cfg.seed, cfg.ci)
            cells.append(f"{est:.3f} [{low:.2f},{high:.2f}]")
        print(f"  {tlbl:<{lbl_w}}" + "".join(f"{c:>{col_w}}" for c in cells))

    # --- Node detection rate by occlusion tercile ---
    hdr("Node recall by occlusion tercile  (mean over runs)")
    for t, tlbl in zip(THIRD_NAMES, THIRD_LABELS):
        cells = []
        for pp in all_exp.values():
            hits = [h for runs in pp.values() if (h := node_hits(runs, node_tercile, t)) is not None]
            est, low, high = detection_rate_ci(hits, cfg)
            cells.append(f"{100 * est:.1f}% [{100 * low:.0f},{100 * high:.0f}]")
        print(f"  {tlbl:<{lbl_w}}" + "".join(f"{c:>{col_w}}" for c in cells))

    # --- Position error ---
    hdr("Position error of matched nodes  (all runs pooled)")
    for stat in ("median", "mean"):
        vals = []
        for pp in all_exp.values():
            dists = [r.m_dist for runs in pp.values() for r in runs if len(r.m_dist)]
            d = np.concatenate(dists) if dists else np.array([np.nan])
            vals.append(float(np.median(d)) if stat == "median" else float(np.mean(d)))
        row(f"Pos error {stat} (m)", vals, fmt=lambda v: f"{v:.4f}")

    print()


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main(gt: Path, output: Path, experiments: Dict[str, Path],
         distance_threshold: float, yaw_threshold: float,
         pitch_threshold: float, orientation_check: bool,
         class_agnostic_fallback: bool = False,
         band: str = "quantile", bootstrap: int = 2000, ci: float = 0.95,
         boot_seed: int = 0, viz_all_runs: bool = False, paired_test: bool = False,
         recall_threshold: float = 0.30) -> None:

    cfg = BandConfig(kind=band, n_boot=bootstrap, ci=ci, seed=boot_seed)

    print("=" * 60)
    print("World Model Evaluation")
    print(f"  GT folder:   {gt}")
    print(f"  Output dir:  {output}")
    print(f"  Experiments: {', '.join(experiments.keys())}")
    print(f"  dist={distance_threshold}m  yaw={yaw_threshold}°  "
          f"pitch={pitch_threshold}°  orientation_check={orientation_check}  "
          f"class_agnostic_fallback={class_agnostic_fallback}")
    print(f"  band={cfg.label()}  bootstrap={cfg.n_boot}  seed={cfg.seed}")
    print("=" * 60)

    gt_dict = load_gt(gt)
    if not gt_dict:
        raise ValueError(f"No GT CSV files found in {gt}")

    node_tercile, plant_cmp_tercile, plant_cmp_value, nq1, nq2 = \
        compute_tercile_maps(gt_dict)
    print(f"\nNode occlusion tercile cutoffs: q1={nq1:.4f}, q2={nq2:.4f}")
    for t in THIRD_NAMES:
        print(f"  {t}: {sum(v == t for v in node_tercile.values())} nodes")

    all_exp: Dict[str, Dict[str, List[PlantResult]]] = {}

    for exp_name, pred_folder in experiments.items():
        print(f"\n-- Experiment: {exp_name} --")
        if not pred_folder.exists():
            raise FileNotFoundError(f"Prediction folder not found: {pred_folder}")

        per_plant = run_experiment(pred_folder, gt_dict,
                                   distance_threshold, yaw_threshold,
                                   pitch_threshold, orientation_check,
                                   class_agnostic_fallback)
        all_exp[exp_name] = per_plant

        vals = [r.vp_metrics[-1] for runs in per_plant.values() for r in runs if len(r.vp_metrics)]
        if vals:
            vm = np.array(vals)
            print(f"  Mean final over all runs  precision={vm[:,0].mean():.3f}  "
                  f"recall={vm[:,1].mean():.3f}  F1={vm[:,2].mean():.3f}")

        exp_out = output / exp_name
        tsv_dir = exp_out / "node_matches"
        print(f"  Saving TSVs > {tsv_dir}")
        several_runs = any(len(runs) > 1 for runs in per_plant.values())
        for pid, runs in per_plant.items():
            for r in runs:
                name = f"{pid}_run{r.run:03d}" if several_runs else pid
                save_node_matches_tsv(name, r, tsv_dir,
                                      plant_cmp_value.get(pid, float('nan')),
                                      plant_cmp_tercile.get(pid, ''))
        save_metrics_summary_tsv(per_plant, exp_out / "metrics_summary.tsv",
                                 plant_cmp_value, plant_cmp_tercile)
        save_metrics_per_plant_tsv(per_plant, exp_out / "metrics_summary_per_plant.tsv",
                                   plant_cmp_value, plant_cmp_tercile)

        viz_dir = exp_out / "3d_visualizations"
        print(f"  Saving 3D visualizations > {viz_dir}"
              f"{'' if viz_all_runs else ' (first run only)'}")
        for pid, runs in per_plant.items():
            for r in (runs if viz_all_runs else runs[:1]):
                name = f"{pid}_run{r.run:03d}" if (several_runs and viz_all_runs) else pid
                try:
                    save_3d_visualization(name, r, viz_dir)
                except Exception as e:
                    print(f"    Warning: 3D vis failed for {name}: {e}")

    save_summary_table_tsv(all_exp, plant_cmp_tercile, output / "summary_table.tsv", cfg)
    print(f"\nSaved the table of the paper > {output / 'summary_table.tsv'}")
    if len(all_exp) > 1:
        save_paired_comparison_tsv(all_exp, plant_cmp_tercile, output / "paired_comparison.tsv",
                                   cfg, with_test=paired_test)
        print(f"Saved the paired comparison > {output / 'paired_comparison.tsv'}")
    vp_tsv = output / f"viewpoints_to_{round(recall_threshold * 100)}pct.tsv"
    save_viewpoints_to_threshold_tsv(all_exp, plant_cmp_tercile, vp_tsv, cfg, recall_threshold)
    print(f"Saved the viewpoints to {round(recall_threshold * 100)}% > {vp_tsv}")
    save_curves_tsv(all_exp, plant_cmp_tercile, output / "curves_by_complexity.tsv", cfg)
    print(f"Saved the curves > {output / 'curves_by_complexity.tsv'}")

    fig_dir = output / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n-- Generating figures > {fig_dir} --")

    fig_avg_prec_rec_f1(all_exp, fig_dir, cfg)
    fig_avg_by_complexity_third(all_exp, plant_cmp_tercile, fig_dir, cfg)
    fig_node_recall_pct_by_third(all_exp, node_tercile, fig_dir, cfg)
    fig_node_error_boxplots(all_exp, node_tercile, fig_dir)
    fig_node_recall_vs_occlusion_binned(all_exp, gt_dict, fig_dir)
    fig_viewpoints_to_30pct_by_complexity(all_exp, plant_cmp_tercile, fig_dir, cfg, recall_threshold)
    for exp_name, per_plant in all_exp.items():
        fig_individual_by_complexity_third(exp_name, per_plant, plant_cmp_tercile, fig_dir)

    print("\n-- Done. --")

    print_results_table(all_exp, node_tercile, plant_cmp_tercile, cfg)
    print_paired_comparison(all_exp, cfg)


def cli(argv=None) -> None:
    p = argparse.ArgumentParser(description="World Model Evaluation",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--experiments', nargs='+', required=True, metavar='NAME:PATH')
    p.add_argument('--gt',     required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--distance_threshold', type=float, default=0.05)
    p.add_argument('--yaw_threshold',      type=float, default=20.0)
    p.add_argument('--pitch_threshold',    type=float, default=30.0)
    p.add_argument('--orientation_check',         action='store_true', default=False)
    p.add_argument('--class_agnostic_fallback', action='store_true', default=False)
    p.add_argument('--band', choices=['quantile', 'supt', 'chi2', 'pointwise', 'iqr'], default='quantile',
                   help='band in the curve figures (default: quantile, simultaneous)')
    p.add_argument('--bootstrap', type=int, default=2000, help='bootstrap resamples')
    p.add_argument('--ci', type=float, default=0.95, help='confidence level')
    p.add_argument('--boot_seed', type=int, default=0, help='seed of the bootstrap')
    p.add_argument('--viz_all_runs', action='store_true', default=False,
                   help='3D visualisations for every run instead of the first one')
    p.add_argument('--paired_test', action='store_true', default=False,
                   help='add a wilcoxon p value to the paired comparison of every pair')
    p.add_argument('--recall_threshold', type=float, default=0.30,
                   help='recall for the viewpoint efficiency figure and table (default: 0.30)')
    args = p.parse_args(argv)

    exps: Dict[str, Path] = {}
    for spec in args.experiments:
        if ':' not in spec:
            raise ValueError(f"Experiment spec must be 'name:path', got: {spec!r}")
        name, path = spec.split(':', 1)
        exps[name] = Path(path)

    main(gt=args.gt, output=args.output, experiments=exps,
         distance_threshold=args.distance_threshold,
         yaw_threshold=args.yaw_threshold,
         pitch_threshold=args.pitch_threshold,
         orientation_check=args.orientation_check,
         class_agnostic_fallback=args.class_agnostic_fallback,
         band=args.band, bootstrap=args.bootstrap, ci=args.ci,
         boot_seed=args.boot_seed, viz_all_runs=args.viz_all_runs,
         paired_test=args.paired_test, recall_threshold=args.recall_threshold)


if __name__ == "__main__":
    cli()

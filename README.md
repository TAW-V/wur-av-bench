# WUR-AV-Bench: A Benchmark for Active Vision in Complex Greenhouse Environments

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Python package for evaluating 3D and 5D node reconstructions and reconstruction efficiency, annotation tools for 5D node annotation in pointclouds, A visibility calculation tool and pose conversion utilities.

## Related repositories

This repository contains the Task III and IV tools of the WUR-AV-Bench benchmark (multi-view 3D reconstruction and 5D pose evaluation). The image-level evaluation of Tasks I and II (background-aware detection and 2D pose) lives in a separate repository, [wur-av-bench-cocoeval](https://github.com/TAW-V/wur-av-bench-cocoeval), a fork of [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval). Both repositories accompany our paper (citation will be added upon publication).

## Dataset

The ground truth node annotations used by `eval`, the baseline runs, the detection filter regions and the data of Tasks I and II are in [wur-av-bench-dataset](https://github.com/TAW-V/wur-av-bench-dataset). The viewpoint data of the 25 plants (RGB images, point clouds and camera poses) is published separately on 4TU.ResearchData; the dataset repository links to it.

## Installation

```bash
uv run wur-av-bench <subcommand> [args]
```

All tools are available as subcommands of the single `wur-av-bench` entry point:

```
wur-av-bench eval             Evaluate predicted model against ground truth
wur-av-bench annotate         Annotate keypoints on nodes in a point cloud (Open3D GUI) and save as CSV
wur-av-bench convert-pose     Convert edge+pt keypoint pairs to position + quaternion pose CSVs
wur-av-bench convert-axes     Rotate pose CSVs from +Z forward to +X forward (convert Open3D to ROS convention)
wur-av-bench visibility       Calculate occlusion/visibility levels for annotated nodes
wur-av-bench fit-stem-lines   Fit piecewise linear segments to plant stem edge keypoints
```

---

## Coordinate systems

Two right-handed coordinate systems are used across the tools:

| Convention | Forward | Up  | Used by |
|------------|---------|-----|---------|
| Open3D (+Z forward) | +Z | +Y | `annotate`, `convert-pose`, `visibility` |
| ROS (+X forward)    | +X | +Z | `eval`, `fit-stem-lines`, `convert-axes` |

The pipeline used in the paper was:

```
annotate  >  convert-pose  >  visibility  >  convert-axes  >  eval
              (Open3D, +Z)                    (ROS, +X)
```

`convert-axes` applies a −90° rotation about the Y-axis to rotate quaternions from
Open3D convention (+Z forward) to ROS convention (+X forward).  Run it after
`visibility` and before `eval` and `fit-stem-lines`.

---

## wur-av-bench eval

Evaluate node reconstructions against ground-truth annotations.

### Input layout

| Path | Contents |
|------|----------|
| `GT_FOLDER/` | One CSV per plant (e.g. `plant_01.csv`). Each row is an annotated node with columns `x y z qx qy qz qw class occlusion_level`. Poses must be in ROS convention (+X forward) — i.e. output of `wur-av-bench convert-axes`. |
| `PRED_FOLDER/` | One JSON per plant (same stem as the GT CSV), or one JSON per plant and run named `<plant>_run<k>.json`. Each JSON is the predicted model produced by the reconstruction pipeline, also in ROS convention (+X forward). |

### Several runs per plant

A planner that samples viewpoints at random gives a different result every time. Name the files
`plant1_run000.json`, `plant1_run001.json`, ... and every run is evaluated separately. `plant1.json`
without a run number counts as a single run, so deterministic methods keep working as before.

Results are first averaged over the runs of a plant and then summarised over plants, so a number still
describes the typical plant. The uncertainty comes from a hierarchical bootstrap that resamples plants
and, inside every plant, its runs.

The curve figures get a simultaneous confidence band (`--band quantile`, the default): it is made to
contain the whole curve, at all viewpoints at once, with 95% confidence, instead of each viewpoint on its
own. It starts from the bootstrap percentiles at every viewpoint and widens them all by the same amount
until 95% of the bootstrap curves lie entirely inside (Montiel Olea and Plagborg-Moller, 2019, algorithm 2).
So the band can be asymmetric, which fits metrics close to 0 or 1.

### Usage

```bash
wur-av-bench eval \
    --gt      /path/to/gt_folder \
    --output  /path/to/output_dir \
    --experiments Active:/path/to/active/preds Random:/path/to/random/preds
```

The `--experiments` flag accepts one or more `NAME:PATH` pairs. Multiple experiments
are evaluated side-by-side and a summary table is printed at the end.

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--gt PATH` | *(required)* | Ground-truth folder (one CSV per plant). |
| `--output PATH` | *(required)* | Output directory; figures, TSVs and 3-D visualisations are written here. |
| `--experiments NAME:PATH …` | *(required)* | One or more experiment name + prediction-folder pairs. |
| `--distance_threshold M` | `0.05` | Maximum Euclidean distance (metres) to count a match. |
| `--yaw_threshold DEG` | `20.0` | Yaw tolerance when orientation gating is enabled. |
| `--pitch_threshold DEG` | `30.0` | Pitch tolerance when orientation gating is enabled. |
| `--orientation_check` | off | Gate matching on yaw/pitch in addition to distance. |
| `--class_agnostic_fallback` | off | Enable the class-agnostic third matching pass. |
| `--band KIND` | `quantile` | Band in the curve figures: `quantile`, `supt`, `chi2`, `pointwise` or `iqr`. |
| `--bootstrap N` | `2000` | Bootstrap resamples for the intervals. |
| `--ci FRACTION` | `0.95` | Confidence level. |
| `--boot_seed N` | `0` | Seed of the bootstrap, so figures are reproducible. |
| `--viz_all_runs` | off | Write 3D visualisations for every run instead of the first one. |
| `--paired_test` | off | Add a Wilcoxon signed-rank p value to the paired comparison. |
| `--recall_threshold F` | `0.30` | Recall for the viewpoint efficiency figure and table, e.g. `0.5` for 50%. |

### Outputs

```
output_dir/
├── summary_table.tsv                  median over plants with interval, per metric and tercile
├── paired_comparison.tsv              every pair of experiments compared per plant (two or more experiments)
├── viewpoints_to_30pct.tsv            numbers behind the viewpoints to 30% figure (name follows --recall_threshold)
├── curves_by_complexity.tsv           precision, recall and F1 per number of viewpoints, with band
├── figures/
│   ├── avg_precision_recall_f1_over_vp.pdf
│   ├── avg_precision_recall_f1_over_vp_by_complexity.pdf
│   ├── node_recall_percentage_by_occlusion_level.pdf
│   ├── node_error_boxplots.pdf
│   ├── node_recall_vs_occlusion_level_binned.pdf
│   ├── viewpoints_to_30pct_by_complexity_boxplot.pdf
│   └── precision_recall_f1_individual_plants_by_complexity_{experiment}.pdf
└── {experiment}/
    ├── metrics_summary.tsv            one row per plant and run
    ├── metrics_summary_per_plant.tsv  mean, sd, min and max over the runs of a plant
    ├── node_matches/
    │   └── {plant_id}[_run{k}].tsv
    └── 3d_visualizations/
        ├── {plant_id}_comparison.obj
        └── {plant_id}_comparison.ply
```

`viewpoints_to_30pct.tsv` gives per experiment and complexity tercile the share of the runs
that reach 30% recall, with its bootstrap interval, and the median and quartiles of the per plant
median number of viewpoints, over the plants where more than half of the runs reach it.
`curves_by_complexity.tsv` gives the lines and bands of the curve figures, with `viewpoints` the
number of viewpoints used (1 is after the first viewpoint).

A compact metrics x experiments summary table is printed to stdout at the end.

---

## wur-av-bench annotate

Interactive Open3D GUI for annotating two keypoints that define a node (petioles / peduncles) in a point cloud.

### Usage

```bash
wur-av-bench annotate path/to/pointcloud.ply [--radius 0.003] [--filter 10.0]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `ply` | *(required)* | Path to the PLY point-cloud file to annotate. |
| `--radius M` | `0.003` | Cylinder radius (metres) used to display annotated nodes. |
| `--filter M` | `10.0` | Keep only points within this distance from the median centre of the pointcloud(outlier removal). |

### CSV format

Each keypoint is a directed cylinder defined by two 3-D points:

| Column | Description |
|--------|-------------|
| `ID` | Integer keypoint identifier |
| `x_edge`, `y_edge`, `z_edge` | Base / edge point of the cylinder (metres) |
| `x_pt`, `y_pt`, `z_pt` | Tip point of the cylinder (metres) |
| `class` | Node class string (e.g. `petiole`, `peduncle`) |

The tool reads `<ply_stem>.csv` on startup (creates an empty one if absent) and saves back
to the same path on Save CSV. Timestamped backups are written to `./bak/`.

### GUI layout

The window is split into a 3D scene (left) and a side panel (right).

**Side-panel buttons (top)**

| Button | Action |
|--------|--------|
| `Save CSV` | Write current keypoints to `<ply_stem>.csv` and create a backup |
| `Set Pt New KP` | Store the current camera pivot* as the *tip* of the next keypoint |
| `Set Edge New KP` | Store the current camera pivot* as the *edge* of the next keypoint |
| `Add KP` | Create a new keypoint (uses stored pivot pair if both are set, otherwise camera pivot) |
| `Re-ID` | Renumber all keypoints 1…N sorted by edge Z-coordinate |

* The camera pivot can be set by double-clicking in the 3D scene.

**Per-keypoint row**

Each keypoint gets one row with:
- Numeric editors for `pt(x/y/z)` and `edge(x/y/z)` with `+` / `-` step buttons (1 mm per click)
- `E-` / `E+` — move the edge point along the cylinder axis (5 % of length per click)
- `P-` / `P+` — move the tip point along the cylinder axis
- A text field for the class name
- `x` — delete the keypoint (confirmation dialog)

---

## wur-av-bench convert-pose

Convert raw node annotations (edge + pt cylinder endpoints) into pose CSVs (position + quaternion).

The direction vector `pt − edge` is converted to a quaternion that rotates `+X` onto that vector.

### Usage

```bash
wur-av-bench convert-pose INPUT_DIR [OUTPUT_DIR] [--quat-order {xyzw,wxyz}]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | *(required)* | Directory containing source `*.csv` files. |
| `OUTPUT_DIR` | `INPUT_DIR/poses` | Destination directory for converted CSVs. |
| `--quat-order` | `xyzw` | Quaternion column order: `xyzw` or `wxyz`. |

### Output columns

`ID, x, y, z` + quaternion columns, plus every other original column (e.g. `class`, `visibility`).
The raw coordinate columns (`x_edge, y_edge, z_edge, x_pt, y_pt, z_pt`) are removed.

---

## wur-av-bench convert-axes

Rotate the quaternions in pose CSVs from Open3D convention (`+Z` forward) to ROS convention
(`+X` forward), by applying a −90° rotation about the Y-axis.

### Usage

```bash
wur-av-bench convert-axes INPUT_DIR OUTPUT_DIR
```

| Argument | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | *(required)* | Directory containing source `*.csv` files with `qx qy qz qw` columns. |
| `OUTPUT_DIR` | *(required)* | Destination directory for converted CSVs. |

### Transformation

```
+Z forward  >  +X forward
+X right    >  +Y right
+Y up       >  +Z up
```

---

## wur-av-bench visibility

Calculate the occlusion / visibility level for each annotated node by rendering its
cylinder from many sampled camera positions and measuring what fraction of the cylinder
is visible (not occluded by the surrounding point cloud).

The result is stored as a new `occlusion_level` column in the output CSV.

### Usage

Single file:
```bash
wur-av-bench visibility --input INPUT.csv --ply-folder PLY_DIR --output OUTPUT.csv
```

Directory (all `*.csv` in `INPUT_DIR`):
```bash
wur-av-bench visibility --input-dir INPUT_DIR --ply-folder PLY_DIR --output-dir OUTPUT_DIR
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `--input-dir` | *(required)* | Input CSV file or directory of CSVs. |
| `--ply-folder` | *(required)* | Directory containing `*.ply` point-cloud files (stem must match CSV stem). |
| `--output` / `--output-dir` | *(required)* | Output CSV file or directory. |
| `--visibility-threshold` | `0.5` | Fraction of cylinder pixels below which a pose counts as occluded. |
| `--num-poses` | `1000` | Number of camera poses to sample per node. |
| `--max-node-distance` | `0.49` | Crop radius (m) around the node applied to the point cloud before rendering. |
| `--num-processes` | auto | Worker processes for parallel rendering (default: `min(cpu_count−2, num_poses/4)`). |
| `--camera-radius` | `0.4` | Distance from the node center to the sampled camera positions (m). |
| `--min-polar-angle` | `-15` | Lower bound of the polar (vertical) sampling range (degrees). |
| `--max-polar-angle` | `15` | Upper bound of the polar (vertical) sampling range (degrees). |
| `--min-azimuth-angle` | `45` | Lower bound of the azimuth (horizontal) sampling range (degrees). |
| `--max-azimuth-angle` | `135` | Upper bound of the azimuth (horizontal) sampling range (degrees). |
| `--noise-xy` | `0.05` | Random x/y offset applied to each camera position to break symmetry (m). |
| `--save-debug-plots` | off | Save one debug plot per node (last pose of each batch). |
| `--always-save-debug-plots` | off | Save a debug plot for every pose. |
| `--save-original-color-plots` | off | Include the original-colour point cloud in debug plots. |

### Input CSV columns

The input CSV must contain `ID, x, y, z, qx, qy, qz, qw` (pose in Open3D / +Z forward convention,
as produced by `wur-av-bench convert-pose`). To convert to +X forward / ROS convention, run `wur-av-bench convert-axes` after calculating node visibility.

### Output

The input CSV is written to `OUTPUT` with an additional `occlusion_level` column
(float 0-1; 0 = fully occluded, 1 = fully visible).

Debug plots (if enabled) are saved to `OUTPUT_DIR/debug_plots/`.

---

## wur-av-bench fit-stem-lines

Fit three connected piecewise linear segments (bottom / center / top) to the edge
keypoints of ground-truth CSVs.  The segments are optimised simultaneously so that
continuity is enforced by construction (end of bottom = start of center, etc.).

The output `detection_filter_params.json` contains the fitted segment parameters for
each plant and can be used as input to detection filter pipelines.

### Usage

```bash
wur-av-bench fit-stem-lines GT_FOLDER [--output OUTPUT_DIR]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `GT_FOLDER` | *(required)* | Folder containing ground-truth CSV files (one per plant, output of `wur-av-bench convert-axes`). |
| `--output OUTPUT_DIR` | `fit_stem_lines_output` | Output directory for `detection_filter_params.json`. |
| `--radius M` | `0.1` | Cylinder radius (metres) written to the JSON. |
| `--extension M` | `0.04` | Length added to each segment end to extend the cylinder endpoints (metres). |

### Input CSV columns

Only `x, y, z` (node position in metres) are required.  All other columns (`qx qy qz qw`, `class`, `occlusion_level`, etc.) are ignored if present.

### Output

```
OUTPUT_DIR/
└── detection_filter_params.json
```

Each plant is a list of three cylinder dicts (bottom > center > top):

```json
{
  "<plant_id>": [
    { "pointa": [x, y, z], "pointb": [x, y, z], "radius": 0.1 },
    { "pointa": [x, y, z], "pointb": [x, y, z], "radius": 0.1 },
    { "pointa": [x, y, z], "pointb": [x, y, z], "radius": 0.1 }
  ]
}
```

`pointa` and `pointb` are the segment start/end points extended slightly beyond the fitted range (by `--extension`) to ensure detection coverage at the boundaries.

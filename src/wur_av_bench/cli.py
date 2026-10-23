#!/usr/bin/env python3
"""
Entry point for all wur-av-bench subcommands.

Dispatches to the appropriate module based on the first positional argument.

Usage
-----
    wur-av-bench <subcommand> [args …]

Subcommands
-----------
eval            Evaluate predicted world-model graphs against ground truth.
annotate        Interactive Open3D GUI for annotating keypoints in a point cloud.
convert-pose    Convert node annotations (edge+pt) to pose CSVs (position + quaternion).
convert-axes    Rotate pose CSVs from +Z forward (Open3D) to +X forward (ROS) convention.
visibility      Calculate occlusion/visibility levels for annotated nodes.

Run ``wur-av-bench <subcommand> --help`` for per-command options.

Author : Thomas Versmissen <thomas.versmissen@wur.nl>
License: MIT - see LICENSE for full text
Version: 1.0.0
Created: 2025-03-11
"""
from __future__ import annotations

import argparse
import sys


_COMMANDS = {
    "eval":            ("wur_av_bench.eval",                   "cli"),
    "annotate":        ("wur_av_bench.node_annotator",          "main"),
    "convert-pose":    ("wur_av_bench.convert_pose",            "main"),
    "convert-axes":    ("wur_av_bench.convert_axes",            "main"),
    "visibility":      ("wur_av_bench.visibility_calculator",   "main"),
    "fit-stem-lines":  ("wur_av_bench.fit_stem_lines",          "main"),
}

_HELP = {
    "eval":           "Evaluate predicted world-model graphs against ground truth.",
    "annotate":       "Interactive Open3D GUI for annotating keypoints in a point cloud.",
    "convert-pose":   "Convert node annotations (edge+pt) to pose CSVs (position + quaternion).",
    "convert-axes":   "Rotate pose CSVs from +Z forward (Open3D) to +X forward (ROS) convention.",
    "visibility":     "Calculate occlusion/visibility levels for annotated nodes using point cloud data.",
    "fit-stem-lines": "Fit piecewise linear segments to plant stem edge keypoints.",
}


def main() -> None:
    p = argparse.ArgumentParser(
        prog="wur-av-bench",
        description="AgriFoodTEF wur-av-bench tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {cmd:<16} {desc}" for cmd, desc in _HELP.items()),
    )
    p.add_argument("command", choices=list(_COMMANDS), metavar="command",
                   help="{" + ", ".join(_COMMANDS) + "}")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the sub-command")

    ns = p.parse_args()

    module_name, func_name = _COMMANDS[ns.command]
    import importlib
    mod = importlib.import_module(module_name)
    getattr(mod, func_name)(ns.args or None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Interactive Open3D GUI for annotating directed keypoints (petioles / peduncles)
in a plant point cloud.

Each keypoint is represented as a cylinder defined by two 3-D points: an edge
(base) and a pt (tip).  Annotations are loaded from and saved back to a CSV file
with the same stem as the PLY file.

Usage
-----
    wur-av-bench annotate PLY_FILE [--radius M] [--filter M]

Arguments
---------
PLY_FILE        Path to the point-cloud PLY file to annotate.
--radius M      Cylinder display radius in metres (default: 0.003).
--filter M      Keep only points within this distance from the median centre,
                used to remove distant outliers (default: 10.0 m).

CSV format
----------
Columns: ID, x_edge, y_edge, z_edge, x_pt, y_pt, z_pt, class.
The file is read on startup (created empty if absent) and written on Save CSV.
Timestamped backups are placed in ``./bak/``.

Author : Thomas Versmissen <thomas.versmissen@wur.nl>
License: MIT - see LICENSE for full text
"""

from pathlib import Path
import sys
import argparse
import os

import numpy as np
import pandas as pd
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

STEP_RATIO = 0.05  # Relative move for the whole vector
NE_STEP = 1e-3     # Absolute increment per click on the new ± buttons (metres)

# Geometry helper functions

def create_cylinder_between(p1, p2, radius: float = 2e-3, resolution: int = 20,
                             color=(1.0, 0.0, 0.0)) -> o3d.geometry.TriangleMesh:
    """Return a cylinder whose bottom cap is on *p1* and top cap on *p2*."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("p1 and p2 coincide — cylinder length is zero")

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius,
                                                    height=float(length),
                                                    resolution=resolution,
                                                    split=4)
    cyl.paint_uniform_color(color)
    cyl.compute_vertex_normals()

    # Rotate the default cylinder (+Z axis) to align with the direction vector
    z_axis = np.array([0.0, 0.0, 1.0])
    dir_n = direction / length
    cross = np.cross(z_axis, dir_n)
    if np.linalg.norm(cross) >= 1e-6: # Check if vectors are not already (anti-)parallel
        angle = np.arccos(np.clip(np.dot(z_axis, dir_n), -1.0, 1.0))
        axis = cross / np.linalg.norm(cross)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=(0, 0, 0))

    cyl.translate((p1 + p2) / 2.0) # Move cylinder to the midpoint between p1 and p2
    return cyl


def filter_points_by_distance(pcd: o3d.geometry.PointCloud, max_distance: float = 0.8, median_edge: np.ndarray|None = None):
    """Filters points in a point cloud based on their distance from the median center."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    if median_edge is not None:
        # If median_edge is provided, use it as the center
        centre = median_edge
        print(f"Using provided median edge point as center: {centre}, filtering points within {max_distance} m")
    else:
        centre = np.median(pts, axis=0) # Calculate the median center of the points
        print(f"Calculated median center of ply: {centre}, filtering points within {max_distance} m")

    mask = np.linalg.norm(pts - centre, axis=1) <= max_distance # Create a mask for points within max_distance

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(pts[mask])

    if pcd.has_colors():
        new_pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        new_pcd.normals = o3d.utility.Vector3dVector(
            np.asarray(pcd.normals)[mask])

    return new_pcd

# Main viewer class

class KeypointViewer:
    """Open3D window with editable key‑points, edges, and one‑click CSV export."""

    def __init__(self, ply_path: Path, cylinder_radius: float = 2e-3, filter_distance: float = 10.0):
        self.ply_path = Path(ply_path)
        self.cylinder_radius = float(cylinder_radius)
        self.filter_distance = float(filter_distance)
        # self.kps stores: id → {edge, pt, cyl_name, label, ne_edge, ne_pt}
        # ne_edge and ne_pt are NumberEdit widgets, added later by _add_row_ui
        self.kps = {}
        self.pivot = {}
        df = self.initialize_dataframe_from_csv()

        self.step_ratio = STEP_RATIO
        self.ne_step = NE_STEP

        self._build_window(df,name=self.ply_path.name)

    def initialize_dataframe_from_csv(self):
        """Loads keypoint data from a CSV file or initializes an empty DataFrame if not found."""
        self.csv_path = self.ply_path.with_suffix('.csv')
        if self.csv_path.exists():
            print(f"Loading keypoints from {self.csv_path}")
            self.delim = self._detect_delimiter(self.csv_path)
            df = pd.read_csv(self.csv_path, delimiter=self.delim)
        else:
            print(f"CSV file {self.csv_path} does not exist. Creating a new one.")
            # If CSV does not exist, create an empty DataFrame structure
            df = pd.DataFrame(columns=["ID", "x_edge", "y_edge", "z_edge", "x_pt", "y_pt", "z_pt", "class"])
            self.delim = ',' # Default to comma for new files
        if df.empty:
            print(f"CSV file {self.csv_path} is empty or does not exist. "
                  "Please create a CSV file with the required columns.")
            # df = None # Original behavior: set df to None if empty or non-existent
        else:
            # Ensure the CSV has the required columns if it's loaded and not empty
            required_columns = ["ID", "x_edge", "y_edge", "z_edge", "x_pt", "y_pt", "z_pt"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file {self.csv_path} is missing required columns: {required_columns}")
        self.df_backup = df.copy() if df is not None else pd.DataFrame()
        return df

    @staticmethod
    def _detect_delimiter(path: Path) -> str:
        """Detects whether a CSV file uses comma or semicolon as a delimiter by checking first few lines."""
        counts = {",": 0, ";": 0}
        with path.open() as f:
            for _ in range(5): # Check the first 5 lines
                line = f.readline()
                if not line:
                    break
                counts[","] += line.count(",")
                counts[";"] += line.count(";")
        return "," if counts[","] >= counts[";"] else ";"


    def _build_window(self,df: pd.DataFrame|None = None, name: str = "Node Annotator"):
        """Creates the main application window, scene widget, and side panel proxy."""
        print(f"Building window for {name}")
        self.window = gui.Application.instance.create_window(name, 1920, 1080)
        print(f"Window created")
        self.window.set_on_close(lambda: True) 

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.panel_proxy = gui.WidgetProxy() # A proxy widget to hold the actual panel
        self.window.add_child(self.panel_proxy)

        self.cyl_material = self._make_cylinder_material()

        self._load_data(df) # Load point cloud and CSV data
        self._build_side_panel() # Construct the UI for the side panel
        self._set_layout() # Define how scene and panel are arranged


    def _set_layout(self):
        """Defines the layout for the scene widget and the side panel within the main window."""
        def on_layout(_): # The argument is the layout context, not used here
            content = self.window.content_rect
            panel_w = 1200 # Desired width for the side panel
            self.scene_widget.frame = gui.Rect(content.x, content.y,
                                               content.width - panel_w,
                                               content.height)
            self.panel_proxy.frame = gui.Rect(
                content.get_right() - panel_w, content.y,
                panel_w, content.height)
        self.window.set_on_layout(on_layout)
        self.window.add_child(self.scene_widget)

    # Scene and data loading helpers
    def _make_cylinder_material(self):
        """Creates a rendering material for the cylinders representing keypoints."""
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit" # Unlit shader for consistent color
        mat.base_color = (1.0, 0.0, 0.0, 1.0) # Red color
        return mat

    def _load_data(self,df: pd.DataFrame|None = None):
        """Loads the point cloud from PLY and keypoints from the (optional) DataFrame."""

        # Load CSV data if provided
        if df is not None:
            try:
                self.populate_scene_from_csv(df)
            except Exception as e:
                print(f"Error loading CSV data: {e}")

        # Calculate median x y and z of the kps
        if self.kps:
            median_edge = np.median([kp["edge"] for kp in self.kps.values()], axis=0)
        else:
            median_edge = None

        # Load Point Cloud
        pcd = o3d.io.read_point_cloud(str(self.ply_path))
        if pcd.is_empty():
            raise RuntimeError(f"Point‑cloud is empty: {self.ply_path}")
        pcd = filter_points_by_distance(pcd, self.filter_distance, median_edge) # Filter out distant points

        pc_mat = rendering.MaterialRecord()
        pc_mat.shader = "defaultLit"
        pc_mat.point_size = 3.0

        if pcd.has_colors():
            cols = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(cols[:, [2,1,0]])  # Assuming BGR input, convert to RGB for Open3D
        self.scene_widget.scene.add_geometry("cloud", pcd, pc_mat)

        bbox = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

    def populate_scene_from_csv(self,df: pd.DataFrame) -> None:
        """Reads keypoints from the DataFrame and adds corresponding cylinders and labels to the scene."""
        self.df_backup = df.copy() # Backup the initial state of the DataFrame

        managed_cols = ["ID", "x_edge", "y_edge", "z_edge", "x_pt", "y_pt", "z_pt", "class"]
        for _, row in df.iterrows():
            eid = int(row["ID"])
            edge = np.array([row["x_edge"], row["y_edge"], row["z_edge"]], float)
            pt   = np.array([row["x_pt"],   row["y_pt"],   row["z_pt"]],   float)
            cyl_name = f"cyl_{eid}"
            classname = row.get("class", "unknown")

            cyl = create_cylinder_between(edge, pt, radius=self.cylinder_radius)
            self.scene_widget.scene.add_geometry(cyl_name, cyl, self.cyl_material)

            # Position label slightly above the cylinder's midpoint for better visibility
            label_pos = (edge + pt)/2 + np.array([0, 0, self.cylinder_radius * 2])
            label = self.scene_widget.add_3d_label(label_pos.astype(np.float32).reshape(3, 1), str(eid))
            label.scale = 2.0

            extra = {col: row[col] for col in row.index if col not in managed_cols}
            self.kps[eid] = {"edge": edge, "pt": pt, "cyl_name": cyl_name, "label": label, "class": classname, "extra": extra}

    # Side‑panel UI construction
    def _build_side_panel(self):
        """Builds or rebuilds the entire side panel UI from scratch."""
        # Create a new vertical layout for the panel
        new_panel = gui.Vert(6, gui.Margins(10, 8, 10, 8))
        self.panel = new_panel # Store a reference to the current panel being built

        self._add_header_and_buttons() # Add global buttons and column headers

        # Add a row of controls for each keypoint, sorted by ID
        for eid in sorted(self.kps.keys()):
            self._add_row_ui(eid)

        # Set the newly built panel as the content of the panel_proxy
        self.panel_proxy.set_widget(new_panel)

    def _add_header_and_buttons(self):
        """Adds header (Save, Add, Re-ID buttons) and column labels to the panel."""
        btn_save = gui.Button("Save CSV")
        btn_save.set_on_clicked(self._save_csv)
        self.panel.add_child(btn_save)

        btn_pt_pivot = gui.Button("Set Pt New KP")
        btn_pt_pivot.set_on_clicked(self._set_pt_point_as_new_keypoint)
        self.panel.add_child(btn_pt_pivot)
        
        btn_edge_pivot = gui.Button("Set Edge New KP")
        btn_edge_pivot.set_on_clicked(self._set_edge_point_as_new_keypoint)
        self.panel.add_child(btn_edge_pivot)

        btn_add = gui.Button("Add KP")
        btn_add.set_on_clicked(self._add_keypoint_cb)
        self.panel.add_child(btn_add)

        btn_reid = gui.Button("Re-ID")
        btn_reid.set_on_clicked(self._reassign_ids)
        self.panel.add_child(btn_reid)

        # Horizontal layout for column headers
        hdr = gui.Horiz(4)
        hdr.add_child(gui.Label("ID"))
        for name in ("      pt(x)", "pt(y)", "pt(z)", "ed(x)", "ed(y)", "ed(z)", "class"):
            hdr.add_child(gui.Label(f" {name + ' ' * 27}")) # Padded for alignment
        hdr.add_child(gui.Label(" "))  # Spacer for alignment with action buttons in rows
        self.panel.add_child(hdr)

    def _add_row_ui(self, eid: int):
        """Builds one side-panel row for manipulating a single keypoint."""
        row = gui.Horiz(4) # Horizontal layout for this keypoint's controls
        row.add_child(gui.Label(f"{eid:>3}")) # Display keypoint ID

        pt_editors, edge_editors = [], []
        # Create NumberEdits and +/- buttons for 'pt' and 'edge' coordinates
        for vec in ("pt", "edge"): # Iterate for point (pt) and edge coordinates
            for idx in range(3): # Iterate for x, y, z components
                ne = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                ne.set_value(float(self.kps[eid][vec][idx]))
                ne.set_preferred_width(70)
                ne.set_on_value_changed(self._make_update_cb(eid, vec, idx))

                btn_minus = gui.Button("-")
                btn_plus  = gui.Button("+")
                btn_minus.set_on_clicked(self._make_step_cb(ne, eid, vec, idx, -self.ne_step))
                btn_plus.set_on_clicked(self._make_step_cb(ne, eid, vec, idx, +self.ne_step))

                row.add_child(ne)
                row.add_child(btn_minus)
                row.add_child(btn_plus)

                (pt_editors if vec == "pt" else edge_editors).append(ne)

        # Class name input for this keypoint
        name_field = gui.TextEdit()
        name_field.text_value = str(self.kps[eid].get("class", "unknown"))
        name_field.set_on_value_changed(lambda text, eid=eid: self.kps[eid].update({"class": text}))
        row.add_child(name_field)

        # Buttons for adjusting the whole vector (edge or point) along its direction
        btn_e_in  = gui.Button("E-")
        btn_e_out = gui.Button("E+")
        btn_p_in  = gui.Button("P-")
        btn_p_out = gui.Button("P+")
        btn_e_in.set_on_clicked(self._make_adjust_cb(eid, "edge", +1)) # Move edge point inward
        btn_e_out.set_on_clicked(self._make_adjust_cb(eid, "edge", -1))# Move edge point outward
        btn_p_in.set_on_clicked(self._make_adjust_cb(eid, "pt",  +1))  # Move tip point inward
        btn_p_out.set_on_clicked(self._make_adjust_cb(eid, "pt",  -1)) # Move tip point outward
        # Add the name field and buttons to the row
        for b in (btn_e_in, btn_e_out, btn_p_in, btn_p_out):
            row.add_child(b)


        # Delete button for this keypoint
        btn_del = gui.Button("x")
        btn_del.set_on_clicked(self._make_delete_cb(eid))
        row.add_child(btn_del)

        # Store references to the NumberEdit widgets for this keypoint
        # This allows programmatic updates if coordinates change via other means (e.g., E±/P± buttons)
        self.kps[eid]["ne_pt"]   = pt_editors
        self.kps[eid]["ne_edge"] = edge_editors

        self.panel.add_child(row)

    def _reassign_ids(self):
        """Renumbers all keypoints sequentially from 1, sorted by their Z-coordinate (bottom to top)."""
        if not self.kps:
            return

        # Get current keypoint data (edge and point coordinates) sorted by the Z-coordinate of the 'edge' point
        old_ids = sorted(self.kps.keys(), key=lambda i: self.kps[i]["edge"][2])
        data = [(self.kps[i]["edge"].copy(), self.kps[i]["pt"].copy(), self.kps[i].get("class", "unknown")) for i in old_ids]

        # Clear existing keypoints from the scene and internal storage
        for kp in self.kps.values():
            self._safe_remove_geometry(self.scene_widget.scene, kp["cyl_name"])
            if kp.get("label"):
                try:
                    self.scene_widget.remove_3d_label(kp["label"])
                except Exception: # Catching broad exception as in original
                    pass
        self.kps.clear()

        # Rebuild keypoints with new sequential IDs
        for new_id, (edge, pt, class_name) in enumerate(data, start=1):
            self.kps[new_id] = {
                "edge": edge, "pt": pt,
                "cyl_name": f"cyl_{new_id}", # Name based on new ID
                "label": None, # Label will be (re)created by _update_cylinder_and_label
                "class": class_name
            }
            self._update_cylinder_and_label(new_id) # Add new geometry and label to scene

        # Rebuild the entire side panel to reflect new IDs and order
        self._rebuild_panel()

    def _make_delete_cb(self, eid: int):
        """Creates a callback for a keypoint's delete button, including a confirmation dialog."""
        def _cb():
            dialog = gui.Dialog("Confirm Deletion")
            layout = gui.Vert(10, gui.Margins(10, 10, 10, 10))
            layout.add_child(gui.Label(f"Are you sure you want to delete keypoint {eid}?"))

            button_layout = gui.Horiz(5)
            def confirm_deletion():
                # Remove keypoint from internal storage
                kp = self.kps.pop(eid, None)
                if kp:
                    # Remove associated geometry and label from the scene
                    self._safe_remove_geometry(self.scene_widget.scene, kp["cyl_name"])
                    if kp.get("label") is not None:
                        try:
                            self.scene_widget.remove_3d_label(kp["label"])
                        except AttributeError: # Original code uses AttributeError here
                            pass
                self.window.close_dialog()
                self._rebuild_panel() # Update the UI

            def cancel_deletion():
                self.window.close_dialog()

            btn_ok = gui.Button("Delete")
            btn_ok.set_on_clicked(confirm_deletion)
            btn_cancel = gui.Button("Cancel")
            btn_cancel.set_on_clicked(cancel_deletion)

            button_layout.add_child(btn_ok)
            button_layout.add_child(btn_cancel)
            layout.add_child(button_layout)

            dialog.add_child(layout)
            self.window.show_dialog(dialog)
        return _cb

    def _set_edge_point_as_new_keypoint(self):
        """Saves the current camera pivot point as the edge or point of a new keypoint."""
        pivot = np.asarray(self.scene_widget.center_of_rotation).reshape(-1)
        self.pivot["edge"] = pivot # Store the pivot point for later use

    def _set_pt_point_as_new_keypoint(self):
        """Saves the current camera pivot point as the point of a new keypoint."""
        pivot = np.asarray(self.scene_widget.center_of_rotation).reshape(-1)
        self.pivot["pt"] = pivot

    def _add_keypoint_cb(self):
        """Callback for 'Add KP' button. Creates a new keypoint at the current camera pivot point."""
        # Determine the next available ID
        new_id = (max(self.kps) + 1) if self.kps else 1

        # Default setup for edge and pt
        edge, pt = None, None
        
        # Check if both edge and pt pivots are set
        if "edge" in self.pivot and "pt" in self.pivot:
            # Use the stored pivot points if they're different
            if not np.array_equal(self.pivot["edge"], self.pivot["pt"]):
                edge = self.pivot["edge"].copy()
                pt = self.pivot["pt"].copy()
                self.pivot.clear()  # Clear pivots after use
        
        # If edge and pt weren't both set or were identical, use camera pivot
        if edge is None or pt is None:
            pivot = np.asarray(self.scene_widget.center_of_rotation).reshape(-1)
            edge = pivot.copy()
            pt = pivot + np.array([5e-3, 5e-3, 5e-3])  # Small offset for initial cylinder visibility

        # Register the new keypoint internally
        self.kps[new_id] = {
            "edge": edge,
            "pt": pt,
            "cyl_name": f"cyl_{new_id}",
            "label": None # Label will be created by _update_cylinder_and_label
        }
        self._update_cylinder_and_label(new_id) # Add geometry and label to scene
        self._rebuild_panel() # Rebuild side panel to include the new keypoint

    def _rebuild_panel(self):
        """Completely rebuilds the side panel UI and signals the window to update its layout."""
        self._build_side_panel()
        self.window.set_needs_layout()
        self.scene_widget.force_redraw()

    # Callbacks and geometry synchronization
    @staticmethod
    def _safe_remove_geometry(scene: rendering.Open3DScene, name: str):
        """Safely removes a geometry item from the scene"""
        scene.remove_geometry(name)

    def _update_cylinder_and_label(self, eid: int):
        """Updates or creates the cylinder geometry and 3D label for a given keypoint ID."""
        kp = self.kps[eid]
        self._safe_remove_geometry(self.scene_widget.scene, kp["cyl_name"]) # Remove old cylinder if it exists
        try:
            new_cyl = create_cylinder_between(kp["edge"], kp["pt"], radius=self.cylinder_radius)
        except ValueError:
            return  # Ignore zero‑length edge during editing (p1 and p2 coincide)
        self.scene_widget.scene.add_geometry(kp["cyl_name"], new_cyl, self.cyl_material)

        # Update or create the 3D label
        if kp.get("label") is not None:
            try:
                self.scene_widget.remove_3d_label(kp["label"]) # Remove old label
            except AttributeError: 
                pass
        pos = (kp["edge"] + kp["pt"]) / 2 + np.array([0, 0, self.cylinder_radius * 2]) # Position label above cylinder
        kp["label"] = self.scene_widget.add_3d_label(pos.astype(np.float32).reshape(3, 1), str(eid))
        kp["label"].scale = 2.0

    # Callback for NumberEdit direct value change
    def _make_update_cb(self, eid: int, vec: str, idx: int):
        """Creates a callback for when a NumberEdit's value is changed by typing."""
        def _cb(new_val: float):
            kp = self.kps[eid]
            kp[vec][idx] = float(new_val) # Update internal coordinate
            self._update_cylinder_and_label(eid) # Update scene geometry
        return _cb

    # Callback for per-coordinate +/- buttons
    def _make_step_cb(self, ne, eid: int, vec: str, idx: int, delta: float):
        """Creates a callback for the +/- buttons associated with a NumberEdit."""
        def _cb():
            # Read current value from the NumberEdit widget
            cur = ne.double_value
            new_val = cur + delta

            # Write the new value back to the NumberEdit widget
            ne.set_value(new_val)

            # Update internal coordinate and scene geometry
            kp = self.kps[eid]
            kp[vec][idx] = new_val
            self._update_cylinder_and_label(eid)
        return _cb

    # Callback for vector-level E±/P± buttons
    def _make_adjust_cb(self, eid: int, vec: str, direction: int):
        """Creates a callback for E±/P± buttons, moving 'edge' or 'pt' along the vector between them."""
        def _cb():
            kp = self.kps[eid]
            edge, pt = kp["edge"], kp["pt"]
            if vec == "edge":
                v = pt - edge # Vector from edge to pt
            else: # vec == "pt"
                v = edge - pt # Vector from pt to edge
            length = np.linalg.norm(v)
            if length == 0: # Avoid division by zero if points coincide
                return

            delta_vec = v / length * length * self.step_ratio * direction # Calculate displacement vector
            if vec == "edge":
                new_edge = edge + delta_vec
                kp["edge"] = new_edge
                # Update corresponding NumberEdit widgets in the UI
                for i, val in enumerate(new_edge):
                    kp["ne_edge"][i].set_value(val)
            else: # vec == "pt"
                new_pt = pt + delta_vec
                kp["pt"] = new_pt
                # Update corresponding NumberEdit widgets in the UI
                for i, val in enumerate(new_pt):
                    kp["ne_pt"][i].set_value(val)
            self._update_cylinder_and_label(eid) # Update scene geometry
        return _cb

    # CSV saving functionality
    def _save_csv(self):
        """Saves the current keypoint data to the CSV file and creates a backup of the original."""
        managed_cols = ["ID", "x_edge", "y_edge", "z_edge", "x_pt", "y_pt", "z_pt", "class"]
        all_cols = list(self.df_backup.columns)
        rows = []
        for eid, data in sorted(self.kps.items()):
            row = {col: None for col in all_cols}
            # Fill managed columns
            row["ID"] = eid
            row["x_edge"], row["y_edge"], row["z_edge"] = data["edge"]
            row["x_pt"], row["y_pt"], row["z_pt"] = data["pt"]
            row["class"] = data.get("class", "unknown")
            # Fill extra columns from original CSV
            if "extra" in data:
                for col, val in data["extra"].items():
                    row[col] = val

            rows.append(row)
        df_out = pd.DataFrame(rows, columns=all_cols)
        out_path = self.csv_path

        # Save the CSV file
        try:
            df_out.to_csv(out_path, sep=self.delim, index=False)
            print(f"Saved {len(rows)} rows to {out_path}")
        except:
            print(f"Error saving CSV to {out_path}")
            errorcsv_dialog = gui.Dialog("Error Saving CSV")
            error_layout = gui.Vert(10, gui.Margins(10, 10, 10, 10))
            error_layout.add_child(gui.Label(f"Failed to save CSV to {out_path}. "
                                              "Please check file permissions or path validity."))
            error_button = gui.Button("OK")
            error_button.set_on_clicked(lambda: self.window.close_dialog())

            error_layout.add_child(error_button)
            errorcsv_dialog.add_child(error_layout)
            self.window.show_dialog(errorcsv_dialog)

        # Create a backup of the original CSV file

        # append datetime to the backup file and change suffix to .bak
        date_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename_without_suffix = self.csv_path.stem
        backup_filename = f"{filename_without_suffix}_{date_str}.bak"
        backup_dir = Path("./bak")
        backup_dir.mkdir(parents=True, exist_ok=True)
        bak = self.csv_path.parent / backup_dir / backup_filename
        print(f"Creating backup {backup_filename} in subfolder {backup_dir}")
        try:
            self.df_backup.to_csv(bak, sep=self.delim, index=False, mode='a')
        except Exception as e:
            print(f"Error creating backup CSV: {e}")
            print(f"Backup not created.")

# Command-line interface setup
def _parse_args(argv):
    """Parses command-line arguments for the script."""
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument("ply", type=Path, help="Path to point‑cloud PLY file")
    p.add_argument("--radius", type=float, default=0.003, help="Cylinder radius in metres (default: 0.003)")
    p.add_argument("--filter", type=float, default=10.0, help="Maximum distance from the median center to keep points (default: 10.0 m)")
    return p.parse_args(argv)

def main(argv=None):
    # Workaround for potential Wayland issue with Open3D GUI
    print("If running under wayland fails with error 'libdecor-gtk-WARNING: Failed to initialize GTK', run $ export XDG_SESSION_TYPE=x11")
    print("Trying to set XDG_SESSION_TYPE=x11 automatically for Open3D GUI compatibility if WAYLAND_DISPLAY is detected...")
    if "WAYLAND_DISPLAY" in os.environ:
        print("Running under Wayland, setting XDG_SESSION_TYPE=x11 (if not already set) for Open3D GUI compatibility.")
        os.environ["XDG_SESSION_TYPE"] = "x11"

    args = _parse_args(argv or sys.argv[1:])
    app = gui.Application.instance
    app.initialize()   
    viewer = KeypointViewer(args.ply, cylinder_radius=args.radius, filter_distance=args.filter)
    app.run()  

if __name__ == "__main__":
    main()
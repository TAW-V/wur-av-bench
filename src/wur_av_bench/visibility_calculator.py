#!/usr/bin/env python3
"""
Calculate the occlusion / visibility level for each annotated node by rendering
the node's cylinder from many sampled camera positions and measuring how much of
it is occluded by the surrounding point cloud.

For each pose the ratio of visible cylinder pixels to total cylinder pixels is
computed; the mean across all camera poses is stored as ``occlusion_level`` in the
output CSV.

Usage
-----
Single file:
    wur-av-bench visibility --input INPUT.csv --ply-folder PLY_DIR --output OUTPUT.csv

Directory:
    wur-av-bench visibility --input-dir INPUT_DIR --ply-folder PLY_DIR --output-dir OUTPUT_DIR

Arguments
---------
--input / --input-dir       Input CSV file or directory of CSVs.
--ply-folder                Directory containing the ``*.ply`` point-cloud files
                            (one PLY per plant, stem must match the CSV stem).
--output / --output-dir     Output CSV file or directory.
--visibility-threshold      Fraction threshold below which a pose is counted as
                            occluded (default: 0.5).
--num-poses                 Number of camera poses to sample per node (default: 1000).
--max-node-distance         Radius (m) around the node used to crop the point cloud
                            (default: 0.49).
--num-processes             Worker processes for parallel rendering (default: auto).

Camera sampling options:
    --camera-radius, --min-polar-angle, --max-polar-angle,
    --min-azimuth-angle, --max-azimuth-angle, --noise-xy

Debug plot options:
    --save-debug-plots, --always-save-debug-plots, --save-original-color-plots

Input CSV format
----------------
The input CSV must contain the following columns (Open3D / +Z forward convention,
as produced by ``wur-av-bench convert-pose``):

    ID          Integer node identifier
    x, y, z     Node position in metres
    qx, qy,     Quaternion orientation (Open3D convention, +Z forward).
    qz, qw      

Output CSV format
-----------------
The input CSV is written to the output path with one additional column appended:

    occlusion_level     Float 0-1.  0 = fully occluded, 1 = fully visible.
                        Computed as the mean visible-pixel fraction across all
                        sampled camera poses.

Author : Thomas Versmissen <thomas.versmissen@wur.nl>
License: MIT - see LICENSE for full text
"""

import numpy as np
import open3d as o3d
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml
import argparse

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def process_csv(input_csv: Path, ply_folder: Path, output_csv: Path, visibility_threshold: float = 0.5, nmbr_poses: int = 1000, max_node_distance: float = 0.49, num_processes: int = 1, occasionally_save_debug_plots: bool = True, always_save_debug_plots: bool = False, save_original_color_plots: bool = False, camera_pose_params: dict|None = None):
    '''
    Process a CSV file containing node poses and visibility information. Ads the visibility level for each node based on the point cloud data.
    Args:
        input_csv (Path): Path to the input CSV file.
        ply_folder (Path): Path to the folder containing the point cloud PLY files.
        output_csv (Path): Path to the output CSV file.
        visibility_threshold (float): Threshold for visibility level.
        nmbr_poses (int): Number of poses to sample for visibility calculation.
        max_node_distance (float): Maximum distance from the node to consider points in the point cloud.
        num_processes (int): Number of processes for parallel processing.
        occasionally_save_debug_plots (bool): Save debug plots occasionally.
        always_save_debug_plots (bool): Always save debug plots.
        save_original_color_plots (bool): Include original colored point cloud in debug plots.
        camera_pose_params (dict): Dictionary containing camera pose sampling parameters:
            - radius (float): Distance from center to camera (default: 0.4)
            - min_polar_angle (float): Minimum polar angle in radians (default: -15 degrees)
            - max_polar_angle (float): Maximum polar angle in radians (default: 15 degrees)
            - min_azimuth_angle (float): Minimum azimuth angle in radians (default: 45 degrees)
            - max_azimuth_angle (float): Maximum azimuth angle in radians (default: 135 degrees)
            - noise_xy (float): Random offset on x/y to break symmetry (default: 0.05)

    CSV format:
        ID, x, y, z, quaternion_x, quaternion_y, quaternion_z, quaternion_w, ......
    '''

    # Set default camera pose parameters if not provided
    if camera_pose_params is None:
        camera_pose_params = {
            'radius': 0.4,
            'min_polar_angle': np.radians(-15),
            'max_polar_angle': np.radians(15),
            'min_azimuth_angle': np.radians(90-45),
            'max_azimuth_angle': np.radians(90+45),
            'noise_xy': 0.05
        }

    # Load the CSV file
    _sep = _detect_delimiter(input_csv)
    df = pd.read_csv(input_csv, sep=_sep)
    if df.empty:
        print(f'\033[91mNo data found in {input_csv}\033[0m')
        return
    
    print(f'Processing {len(df)} nodes from {input_csv}')

    # Check if the required columns are present
    required_columns = ["ID", "x", "y", "z", "qx", "qy", "qz", "qw"]
    if not all(col in df.columns for col in required_columns):
        print(f'\033[91mMissing required columns in {input_csv}\033[0m')
        return
    
    # itterate over each node in the CSV
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing nodes"):
        node_id = int(row["ID"])
        pose = np.array([row["x"], row["y"], row["z"], row["qx"], row["qy"], row["qz"], row["qw"]])

        # Construct the path to the point cloud file. They are located in the parent folder of the CSV file.
        gt_ply = Path( ply_folder, f"{input_csv.stem}.ply")
        
        if not gt_ply.exists():
            print(f'\033[91mPoint cloud file {gt_ply} does not exist for node {node_id}\033[0m')
            continue


        # Sample camera poses
        sampled_poses = _sample_camera_poses(
            center=pose[:3],
            radius=camera_pose_params['radius'],
            n_poses=nmbr_poses,
            min_polar_angle=camera_pose_params['min_polar_angle'],
            max_polar_angle=camera_pose_params['max_polar_angle'],
            min_azimuth_angle=camera_pose_params['min_azimuth_angle'],
            max_azimuth_angle=camera_pose_params['max_azimuth_angle'],
            noise_xy=camera_pose_params['noise_xy']
        )
        # Create new identifiers for sampled poses
        camera_poses = [
            (pose[0], f"{input_csv.stem}_node{node_id}_pose{pose[1]}", pose[2])
            for pose in sampled_poses
        ]

        print(f'Using {len(camera_poses)} camera poses for node {node_id} from {input_csv.stem}')
        
        # Calculate visibility level
        debugplot_folder = output_csv.parent / 'debug_plots'
        if not debugplot_folder.exists():
            debugplot_folder.mkdir(parents=True, exist_ok=True)
        mean_occlusion_lvl, occlusion_lvl_results = calc_occlusion_lvl(pose, gt_ply, input_csv.stem, node_id, camera_poses, visibility_threshold, max_node_distance, str(debugplot_folder), num_processes, occasionally_save_debug_plots, always_save_debug_plots, save_original_color_plots)
        # Add occlusion level to the DataFrame
        df.at[index, 'occlusion_level'] = mean_occlusion_lvl

    # Save the updated DataFrame to the output CSV file
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f'\033[92mProcessed {len(df)} nodes and saved to {output_csv}\033[0m')
    


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
    
def calc_occlusion_lvl(pose:np.ndarray, gt_ply:Path, plant_name:str, node_id:int, camera_poses: list, visibility_threshold:float = 0.5, max_node_distance:float =0.49, debugplot_folder: str = "./occlusion_debug", num_processes: int = 1, occasionally_save_debug_plots: bool = True, always_save_debug_plots: bool = False, save_original_color_plots: bool = False):
    # Load and downsample the point cloud
    ply = o3d.io.read_point_cloud(str(gt_ply))
    
    # remove all points that are more than 50 cm away from the node
    ply_points = np.asarray(ply.points)
    ply_colors = np.asarray(ply.colors)
    distances = np.linalg.norm(ply_points - pose[:3], axis=1)
    ply.points = o3d.utility.Vector3dVector(ply_points[distances < max_node_distance])
    ply.colors = o3d.utility.Vector3dVector(ply_colors[distances < max_node_distance])
    
    if len(ply.points) == 0:
        print(f'\033[91mNo points found for node at position {pose[:3]}\033[0m')
        return None, []

    # Create red cylinder
    cylinder, _c_start, _c_end = _create_cylinder_between(
        pose,
        fixed_length=0.03,
        radius=0.005,
        resolution=20,
        start_offset=0.0,
        color=[1, 0, 0]
    )

    ply = remove_points_near_cylinder(ply, _c_start, _c_end, radius=0.06)  # Remove points near the cylinder axis
    
    # Prepare data for parallel processing, only arrays can be passed to multiprocessing
    ply_points = np.asarray(ply.points)
    ply_colors = np.asarray(ply.colors)
    cylinder_vertices = np.asarray(cylinder.vertices)
    cylinder_triangles = np.asarray(cylinder.triangles)
    
    # Split poses into batches
    batch_size = len(camera_poses) // num_processes
    pose_batches = [camera_poses[i:i+batch_size] for i in range(0, len(camera_poses), batch_size)]
    
    # Prepare arguments for parallel processing
    process_args = [
        (pose_batch, ply_points, ply_colors, cylinder_vertices, cylinder_triangles, plant_name, node_id, visibility_threshold, debugplot_folder, occasionally_save_debug_plots, always_save_debug_plots, save_original_color_plots)
        for i, pose_batch in enumerate(pose_batches)
    ]
    

    # Create a tqdm progress bar for the total number of poses
    pbar = tqdm(total=len(camera_poses), desc="Processing camera poses", ncols=80)
    
    if num_processes == 1:
        # If only one process, process poses sequentially
        batch_results = []
        for args in process_args:
            result = _process_pose_batch(args)
            batch_results.append(result)
            pbar.update(len(result))
    else:
        with Pool(processes=num_processes) as pool:
            batch_results = []
            for result in pool.imap(_process_pose_batch, process_args):
                batch_results.append(result)
                pbar.update(len(result))  # Update progress by the number of poses processed in this batch
        
    pbar.close()
    
    # Flatten results
    occlusion_lvl_results = [result for batch in batch_results for result in batch]
    
    # Compute occlusion level
    mean_node_occlusion_lvl = sum(occlusion_lvl_results) / len(camera_poses)

    return mean_node_occlusion_lvl, occlusion_lvl_results

def remove_points_near_cylinder(pcd, p_start, p_end, radius=0.05):
    """
    Remove points from `pcd` that are within `radius` of the axis between p_start and p_end.
    """
    points = np.asarray(pcd.points)
    direction = p_end - p_start
    height = np.linalg.norm(direction)
    direction = direction / height

    # Vector from p_start to all points
    vec_to_points = points - p_start

    # Project point vectors onto cylinder axis
    projection_lengths = np.dot(vec_to_points, direction)

    # Closest points along the axis
    projections = np.outer(projection_lengths, direction)

    # Closest points on the axis (in 3D space)
    closest_points = p_start + projections

    # Compute distances from points to their axis projections (perpendicular distance)
    distances = np.linalg.norm(points - closest_points, axis=1)

    # Filter logic
    within_radius = distances < radius
    within_height = (projection_lengths >= 0) & (projection_lengths <= height)

    mask = ~(within_radius & within_height)  # Keep points outside the cylinder

    # Apply mask
    filtered_points = points[mask]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # preserve colors if available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors)

    return filtered_pcd

def _create_cylinder_bottom_at_origin(height, radius=0.01, resolution=20):
    """
    Creates a cylinder of given height, oriented along +Z,
    and **ensures** its bottom is at z=0 (rather than -h/2).
    """
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=height,
        resolution=resolution
    )
    # Now fix the position so that the bottom is exactly z=0 in local coords:
    bbox = cylinder.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()  # e.g. [ -r, -r, -h/2 ] if it's centered
    # Shift the geometry so min_z becomes 0
    shift_z = -min_bound[2]
    cylinder.translate([0, 0, shift_z])
    return cylinder

def _create_cylinder_between(
    pose : np.ndarray,
    fixed_length=0.1,
    start_offset=0.0,
    radius=0.01,
    resolution=20,
    color=[1, 0, 0],
):
    """
    Creates a cylinder of `fixed_length` oriented from p1 to p2.
      - With `start_offset=0`, the cylinder's base is exactly at p1.
      - If `start_offset` > 0, the cylinder is moved backwards along its axis.
    
    Returns:
        cylinder: the Open3D mesh
        cyl_start: 3D start point (after offset)
        cyl_end: 3D end point (start + fixed_length * direction)
    """

    # Step 1: Create upright cylinder (z=0 to z=height)
    cylinder = _create_cylinder_bottom_at_origin(
        height=fixed_length,
        radius=radius,
        resolution=resolution
    )
    cylinder.paint_uniform_color(color)

    # Step 2: Rotate to match direction
    position = pose[:3]  # Start point (x, y, z)
    quaternion = pose[3:7]  # Quaternion (x, y, z, w)
    _rot = R.from_quat(quaternion)  # Convert quaternion to rotation matrix
    cylinder.rotate(_rot.as_matrix(), center=np.zeros(3))

    # Step 3: Position the cylinder
    direction = _rot.apply([0, 0, 1])  # Cylinder is oriented along +Z in local coords
    direction = direction / np.linalg.norm(direction)  # Normalize direction vector

    cyl_start = position - direction * start_offset
    cyl_end = cyl_start + direction * fixed_length
    cylinder.translate(cyl_start)

    return cylinder, cyl_start, cyl_end

def _sample_camera_poses(
    center,
    radius=0.5,
    n_poses=10,
    min_polar_angle=np.radians(-30),
    max_polar_angle=np.radians(30),
    min_azimuth_angle=np.radians(75),
    max_azimuth_angle=np.radians(105),
    noise_xy=0.05
):
    """
    Samples camea poses on a spherical cap around a point.

    Args:
        center: 3D position to look at (target point)
        radius: distance from center to camera
        n_poses: number of poses to generate
        angle ranges in radians
        noise_xy: random offset on x/y to break symmetry

    Returns:
        List of 4x4 camera-to-world matrices (numpy arrays)
    """
    poses = []
    cntr = 0
    for _ in range(n_poses):
        # Sample spherical angles
        polar = random.uniform(min_polar_angle, max_polar_angle)     # vertical angle
        azimuth = random.uniform(min_azimuth_angle, max_azimuth_angle)  # horizontal angle

        # Convert to Cartesian coordinates
        x = radius * np.cos(polar) * np.cos(azimuth) + center[0] + random.uniform(-noise_xy, noise_xy)
        y = radius * np.cos(polar) * np.sin(azimuth) + center[1]
        z = radius * np.sin(polar) + center[2]
        cam_pos = np.array([x, y, z])

        # Compute rotation matrix
        rot_mat = _calculate_view_matrix(cam_pos, center)
        rot_mat[:, :2] *= -1

        # Combine into 4x4 pose matrix
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = cam_pos
        poses.append((None, cntr,T))

        cntr += 1

    return poses

def _calculate_view_matrix(cam_pos, target_pos):
    """
    Computes rotation matrix such that camera at `cam_pos` looks at `target_pos`,
    using Open3D convention (+Z = backward, +Y = up).
    """
    cam_pos = np.array(cam_pos)
    target_pos = np.array(target_pos)

    z = (target_pos - cam_pos)
    z /= np.linalg.norm(z)

    # Use world up vector to stabilize the view
    up = np.array([0, 0, 1])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    rot = np.column_stack((x, y, z))

    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1

    return rot

def _process_pose_batch(args):
    poses_batch, ply_points, ply_colors, cylinder_vertices, cylinder_triangles, plantname, node_id, visibility_threshold, debugplot_folder, occasionally_save_debug_plots, always_save_debug_plots, save_original_color_plots = args

    if "WAYLAND_DISPLAY" in os.environ: # fix for Wayland
        os.environ["XDG_SESSION_TYPE"] = "x11"
        
    # Create a new Visualizer for this process
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720) # set to visible=True to see the rendering
    
    try:
        # Recreate Open3D objects from data
        ply = o3d.geometry.PointCloud()
        ply.points = o3d.utility.Vector3dVector(ply_points)
        ply.colors = o3d.utility.Vector3dVector(ply_colors)
        # make ply blue
        ply.paint_uniform_color([0, 0, 0.5])  # Blue color for the point cloud

        
        cylinder = o3d.geometry.TriangleMesh()
        cylinder.vertices = o3d.utility.Vector3dVector(cylinder_vertices)
        cylinder.triangles = o3d.utility.Vector3iVector(cylinder_triangles)
        cylinder.paint_uniform_color([1, 0, 0])
        
        # Add geometries once
        ply_copy = o3d.geometry.PointCloud(ply)
        cylinder_copy = o3d.geometry.TriangleMesh(cylinder)
        
        vis.add_geometry(ply_copy)
        render_option = vis.get_render_option()
        render_option.point_size = 35.0
        render_option.background_color = np.array([0, 0, 0])  # Black background for better contrast
        render_option.light_on = False  # Disable lighting for faster rendering
        
        vis.add_geometry(cylinder_copy)
        
        # Setup camera parameters once
        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam.intrinsic.set_intrinsics(
            width=1280, height=720,
            fx=685*1.2, fy=685*1.2,
            cx=639.5, cy=359.5
        )
        
        batch_results = []
        for pose_idx, camera_pose in enumerate(poses_batch):
            # Update camera pose
            cam.extrinsic = np.linalg.inv(camera_pose[2])
            ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=False)
            
            # Update and render
            vis.poll_events()
            vis.update_renderer()

            # Debugging: print the camera pose
            # dbg = ctr.convert_to_pinhole_camera_parameters()   # read back
            # print("POSE #", pose_idx)
            # print("  sent:\n", np.linalg.inv(camera_pose[2]))
            # print("  kept:\n", dbg.extrinsic)                 # what ViewControl kept

            # Debugging breakpoint
            # if camera_pose[1] == '1723901079720080137':
            #     print('here')
            
            # Get the real image if a path is provided
            real_image = None
            if camera_pose[0] is not None:
                real_image = get_original_image(camera_pose[0], camera_pose[1])
            
            # Capture image with blue point cloud
            rendered_img_cyl_and_ply = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            
            # If original color plots are enabled, capture image with original colors
            rendered_img_original_color = None
            if save_original_color_plots:
                # Restore original colors with smaller point size
                vis.clear_geometries()
                ply_copy = o3d.geometry.PointCloud(ply)
                ply_copy.colors = o3d.utility.Vector3dVector(ply_colors)  # Use original colors
                vis.add_geometry(ply_copy)
                vis.add_geometry(cylinder_copy)
                render_option = vis.get_render_option()
                render_option.point_size = 3.0  # Smaller point size for original colors
                ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=False)
                vis.poll_events()
                vis.update_renderer()
                rendered_img_original_color = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                
                # Reset to blue color and large point size for next iteration
                vis.clear_geometries()
                ply_copy = o3d.geometry.PointCloud(ply)
                ply_copy.paint_uniform_color([0, 0, 0.5])
                vis.add_geometry(ply_copy)
                vis.add_geometry(cylinder_copy)
                render_option = vis.get_render_option()
                render_option.point_size = 15.0  # Reset to large point size for blue points
            
            # --- Render mask image with only the cylinder ---
            vis.clear_geometries()
            cylinder_copy = o3d.geometry.TriangleMesh(cylinder)
            vis.add_geometry(cylinder_copy)
            ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=False)
            vis.poll_events()
            vis.update_renderer()
            renderd_img_cyl_only = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            
            # Create mask and calculate occupancy
            cylinder_mask_img = np.all(renderd_img_cyl_only == [1, 0, 0], axis=-1)
            count_cylinder_px_wo_ply = np.sum(cylinder_mask_img)
            count_cylinder_px_w_ply = np.sum(np.all(rendered_img_cyl_and_ply == [1, 0, 0], axis=-1))

            if count_cylinder_px_w_ply == 0:
                # print(f'\033[91mNo cylinder pixels found in image for pose {pose_idx} of node {node_id}\033[0m')
                visibility_level = 0.0
            elif count_cylinder_px_wo_ply == 0:
                print(f'\033[91mCylinder occluded in image for pose {pose_idx} of node {node_id}\033[0m')
                visibility_level = 0.0
            else:
                visibility_level = count_cylinder_px_w_ply / count_cylinder_px_wo_ply
                
            # Save debug plot
            if (((pose_idx == len(poses_batch) - 1) and occasionally_save_debug_plots) or always_save_debug_plots) and count_cylinder_px_wo_ply > 0:
                save_occlusion_debug_plot(
                    rendered_image=rendered_img_cyl_and_ply,
                    mask_image=cylinder_mask_img,
                    visibility_level=visibility_level,
                    identifier=camera_pose[1],
                    output_folder=debugplot_folder,
                    original_color_image=rendered_img_original_color,
                    real_image=real_image
                )
            
            batch_results.append(visibility_level < visibility_threshold)
            

            #run vis until the user closes it, visibility=True is needed to see the rendering in the window
            #vis.run()

            # Restore geometries for next pose
            vis.clear_geometries()
            vis.add_geometry(ply_copy)
            vis.add_geometry(cylinder_copy)
            
        return batch_results
    finally:
        vis.destroy_window()

def get_original_image(base_folder: str, filename: str) -> np.ndarray | None:
    """
    Load the original color image from the specified base folder.
    The image is expected to be in JPEG format with the same name as the filename.
    """
    image_path = Path(base_folder) / f"{filename}_color.jpeg"

    if image_path.exists():
        return plt.imread(image_path)
    else:
        print(f'\033[91mOriginal color image {image_path} not found\033[0m')
        return None




def save_occlusion_debug_plot(
    rendered_image: np.ndarray,
    mask_image: np.ndarray,
    visibility_level: float,
    identifier: str = "unknown",
    output_folder: str = "./occlusion_debug_all",
    original_color_image: np.ndarray | None = None,
    real_image: np.ndarray | None = None,
):
    os.makedirs(output_folder, exist_ok=True)

    # Determine number of subplots and create figure
    plot_conf = [
        {'title': 'Blue Point Cloud', 'image': rendered_image, 'cmap': None},
        {'title': 'Cylinder Mask', 'image': mask_image, 'cmap': 'gray'},
    ]
    if original_color_image is not None:
        plot_conf.append({'title': 'Original Colors', 'image': original_color_image, 'cmap': None})
    if real_image is not None:
        plot_conf.append({'title': 'Real Image', 'image': real_image, 'cmap': None})

    n_plots = len(plot_conf)
    fig, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1: # Make sure axs is always a list
        axs = [axs]

    # Populate subplots
    for i, conf in enumerate(plot_conf):
        axs[i].imshow(conf['image'], cmap=conf['cmap'])
        axs[i].set_title(conf['title'])
        axs[i].axis("off")

    fig.suptitle(f"Visibility Level: {visibility_level:.2f}", fontsize=14)

    # Construct filename    
    # check if file already exists, if so, increment the number
    filename = f"{identifier}_0.png"
    filepath = os.path.join(output_folder, filename)
    i = 1
    while os.path.exists(filepath):
        filename = f"{identifier}_{i}.png"
        filepath = os.path.join(output_folder, filename)
        i += 1
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="wur-av-bench visibility",
        description="Calculate occlusion/visibility levels for annotated nodes using point cloud data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  wur-av-bench visibility --input data.csv --ply-folder /path/to/plys --output results.csv

  # Directory
  wur-av-bench visibility --input-dir /path/to/csvs --ply-folder /path/to/plys --output-dir /path/to/out

        """
    )

    # Input/Output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=Path, help='Input CSV file path')
    input_group.add_argument('--input-dir', type=Path,
                             help='Directory containing input CSV files (processes all *.csv files)')

    parser.add_argument('--ply-folder', type=Path, required=True,
                        help='Directory containing *.ply point-cloud files (one per plant, stem must match CSV stem)')

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('--output', '-o', type=Path, help='Output CSV file path')
    output_group.add_argument('--output-dir', type=Path, help='Directory for output CSV files')

    # Camera pose arguments
    # Visibility calculation parameters
    parser.add_argument('--visibility-threshold', type=float, default=0.5,
                        help='Threshold for visibility level (default: 0.5)')
    parser.add_argument('--num-poses', type=int, default=1000,
                        help='Number of poses to sample for visibility calculation (default: 1000)')
    parser.add_argument('--max-node-distance', type=float, default=0.49,
                        help='Maximum distance from node to consider points in point cloud (default: 0.49)')

    # Camera pose sampling parameters
    parser.add_argument('--camera-radius', type=float, default=0.4,
                        help='Distance from center to camera for pose sampling (default: 0.4)')
    parser.add_argument('--min-polar-angle', type=float, default=-15,
                        help='Minimum polar angle in degrees (default: -15)')
    parser.add_argument('--max-polar-angle', type=float, default=15,
                        help='Maximum polar angle in degrees (default: 15)')
    parser.add_argument('--min-azimuth-angle', type=float, default=45,
                        help='Minimum azimuth angle in degrees (default: 45)')
    parser.add_argument('--max-azimuth-angle', type=float, default=135,
                        help='Maximum azimuth angle in degrees (default: 135)')
    parser.add_argument('--noise-xy', type=float, default=0.05,
                        help='Random offset on x/y to break symmetry (default: 0.05)')

    # Processing parameters
    parser.add_argument('--num-processes', type=int, default=None,
                        help='Number of processes for parallel processing (default: auto-detect)')

    # Debug plot options
    parser.add_argument('--save-debug-plots', action='store_true',
                        help='Save debug plots occasionally')
    parser.add_argument('--always-save-debug-plots', action='store_true',
                        help='Always save debug plots (overrides --save-debug-plots)')
    parser.add_argument('--save-original-color-plots', action='store_true',
                        help='Include original colored point cloud in debug plots')

    args = parser.parse_args(argv)

    # Set up camera pose parameters
    camera_pose_params = {
        'radius': args.camera_radius,
        'min_polar_angle': np.radians(args.min_polar_angle),
        'max_polar_angle': np.radians(args.max_polar_angle),
        'min_azimuth_angle': np.radians(args.min_azimuth_angle),
        'max_azimuth_angle': np.radians(args.max_azimuth_angle),
        'noise_xy': args.noise_xy
    }

    # Auto-detect number of processes if not specified
    if args.num_processes is None:
        args.num_processes = min(cpu_count()-2, args.num_poses//4)
        args.num_processes = max(args.num_processes, 1)

    # Determine debug plot settings
    occasionally_save_debug_plots = args.save_debug_plots or args.always_save_debug_plots
    always_save_debug_plots = args.always_save_debug_plots

    ply_folder = args.ply_folder.resolve()

    # Process files
    if args.input:
        # Single file processing
        input_csv = args.input.resolve()
        output_csv = args.output.resolve()
        print(f"Processing {input_csv} → {output_csv}")
        process_csv(
            input_csv=input_csv,
            ply_folder=ply_folder,
            output_csv=output_csv,
            visibility_threshold=args.visibility_threshold,
            nmbr_poses=args.num_poses,
            max_node_distance=args.max_node_distance,
            num_processes=args.num_processes,
            occasionally_save_debug_plots=occasionally_save_debug_plots,
            always_save_debug_plots=always_save_debug_plots,
            save_original_color_plots=args.save_original_color_plots,
            camera_pose_params=camera_pose_params
        )
    else:
        # Directory processing
        input_dir = args.input_dir.resolve()
        output_dir = args.output_dir.resolve()

        for csv_file in input_dir.glob("*.csv"):
            input_csv = csv_file.resolve()
            output_csv = output_dir / csv_file.name


            print(f"Processing {input_csv} → {output_csv}")
            process_csv(
                input_csv=input_csv,
                ply_folder=ply_folder,
                output_csv=output_csv,
                visibility_threshold=args.visibility_threshold,
                nmbr_poses=args.num_poses,
                max_node_distance=args.max_node_distance,
                num_processes=args.num_processes,
                occasionally_save_debug_plots=occasionally_save_debug_plots,
                always_save_debug_plots=always_save_debug_plots,
                save_original_color_plots=args.save_original_color_plots,
                camera_pose_params=camera_pose_params
            )


if __name__ == "__main__":
    main()
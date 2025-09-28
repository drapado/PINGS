#!/usr/bin/env python3

import os
import numpy as np
import glob
from pathlib import Path

def create_trajectory_txt():
    """
    Create trajectory.txt files from slam_poses_kitti.txt files in each pings_experiments subfolder.
    Converts lidar poses to camera poses using the transformation from agri_slam dataset.
    """
    
    # Camera to LiDAR transformation (from AgriSLAMDataset)
    T_cam_lidar = np.array([
        [-0.65657749, -0.75423956, -0.00535658, 0.08879001],
        [0.02179564, -0.02607131, 0.99942245, -0.40550301],
        [-0.75394360, 0.65608153, 0.03355697, -0.17441673],
        [0.00000000, 0.00000000, 0.00000000, 1.00000000]
    ])
    
    # Compute lidar to camera transformation (inverse of cam to lidar)
    T_lidar_cam = np.linalg.inv(T_cam_lidar)
    
    # Base directories
    experiments_dir = "pings_experiments"
    agri_data_dir = "agri-data"
    
    # Get all experiment subdirectories
    experiment_folders = [d for d in os.listdir(experiments_dir) 
                         if os.path.isdir(os.path.join(experiments_dir, d))]
    
    for exp_folder in experiment_folders:
        print(f"Processing {exp_folder}...")
        
        # Paths for this experiment
        exp_path = os.path.join(experiments_dir, exp_folder)
        slam_poses_file = os.path.join(exp_path, "slam_poses_kitti.txt")
        
        # Check if slam_poses_kitti.txt exists
        if not os.path.exists(slam_poses_file):
            print(f"  Warning: {slam_poses_file} not found, skipping...")
            continue
            
        # Read SLAM poses (lidar poses in KITTI format)
        try:
            slam_poses = np.loadtxt(slam_poses_file)
            print(f"  Loaded {len(slam_poses)} poses from slam_poses_kitti.txt")
        except Exception as e:
            print(f"  Error loading {slam_poses_file}: {e}")
            continue
            
        # Get timestamps from corresponding agri-data ouster point cloud files
        agri_folder_path = os.path.join(agri_data_dir, exp_folder, "ouster", "points")
        
        if not os.path.exists(agri_folder_path):
            print(f"  Warning: {agri_folder_path} not found, skipping...")
            continue
            
        # Get all .ply files and extract timestamps
        ply_files = sorted(glob.glob(os.path.join(agri_folder_path, "*.ply")))
        
        if len(ply_files) == 0:
            print(f"  Warning: No .ply files found in {agri_folder_path}, skipping...")
            continue
            
        # Extract timestamps from filenames (format: timestamp-nanoseconds.ply)
        timestamps = []
        for ply_file in ply_files:
            filename = os.path.basename(ply_file)
            # Extract timestamp parts (seconds-nanoseconds.ply)
            parts = filename.replace('.ply', '').split('-')
            if len(parts) >= 2:
                try:
                    seconds = int(parts[0])
                    nanoseconds = int(parts[1])
                    # Convert nanoseconds to decimal seconds and combine
                    timestamp = seconds + nanoseconds / 1e9
                    timestamps.append(timestamp)
                except ValueError:
                    print(f"  Warning: Could not parse timestamp from {filename}")
                    continue
            else:
                print(f"  Warning: Unexpected filename format: {filename}")
                continue
                
        print(f"  Found {len(timestamps)} timestamps from .ply files")
        
        # Check if we have matching number of poses and timestamps
        min_count = min(len(slam_poses), len(timestamps))
        if len(slam_poses) != len(timestamps):
            print(f"  Warning: Mismatch between poses ({len(slam_poses)}) and timestamps ({len(timestamps)})")
            print(f"  Using first {min_count} entries")
            
        # Process poses and create output
        trajectory_lines = []
        
        for i in range(min_count):
            # Get KITTI pose (3x4 matrix flattened)
            kitti_pose = slam_poses[i].reshape((3, 4))
            
            # Convert to 4x4 homogeneous matrix (lidar pose)
            T_world_lidar = np.eye(4)
            T_world_lidar[:3, :] = kitti_pose
            
            # Convert lidar pose to camera pose
            # T_world_cam = T_world_lidar * T_lidar_cam
            T_world_cam = T_world_lidar @ T_lidar_cam
            
            # Extract 3x4 transformation matrix for output
            cam_pose_3x4 = T_world_cam[:3, :]
            
            # Format: TIMESTAMP T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23
            timestamp = timestamps[i]
            pose_flat = cam_pose_3x4.flatten()
            
            line = f"{timestamp:.6f}"
            for val in pose_flat:
                line += f" {val:.9f}"
            
            trajectory_lines.append(line)
            
        # Write trajectory.txt file
        output_file = os.path.join(exp_path, "trajectory.txt")
        with open(output_file, 'w') as f:
            for line in trajectory_lines:
                f.write(line + '\n')
                
        print(f"  Created {output_file} with {len(trajectory_lines)} entries")
        
    print("Done!")

if __name__ == "__main__":
    create_trajectory_txt()
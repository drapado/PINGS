# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# 2024 Yue Pan
# 2025 Custom Agri-SLAM Dataset
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
import os
import csv
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PIL import Image


class AgriSLAMDataset:
    def __init__(self, data_dir, sequence: str = "train", *_, **__):
        
        self.contains_image = True
        self.use_sky_removal = True
        # print(data_dir, sequence, "\n\n\n\n\n\n")
        # Hardcoded paths for agri-slam-data/train
        self.agri_slam_dir = f"/packages/pings/agri-data/{data_dir}/"
        
        # Ouster LiDAR point cloud files
        self.ouster_dir = os.path.join(self.agri_slam_dir, "ouster/points/")
        self.scan_files = sorted(glob.glob(self.ouster_dir + "*.ply"))
        scan_count = len(self.scan_files)
        print(f"Found {scan_count} LiDAR scans")
        
        # ZED camera 2 image files
        self.img_dir = os.path.join(self.agri_slam_dir, "zed_multi/cam_2/rgb/")
        self.img_files = sorted(glob.glob(self.img_dir + "*.jpg"))
        img_count = len(self.img_files)
        print(f"Found {img_count} camera images")
        
        # Check if images are available
        if img_count > 0:
            self.image_available = True
            self.load_img = True
        else:
            self.image_available = False
            self.load_img = False
        
        # Build mask paths for sky removal if enabled
        if self.use_sky_removal and self.image_available:
            self.mask_paths = []
            for img_file in self.img_files:
                # Replace /rgb/ with /depth_anything/ and .jpg with .png
                mask_path = img_file.replace('/rgb/', '/depth_anything/').replace('.jpg', '.png')
                self.mask_paths.append(mask_path)
            print(f"Sky removal enabled with depth threshold: 0")

        self.use_only_colorized_points = True
        
        # Load ground truth poses
        self.gt_poses = None
        self.gt_pose_provided = False
        
        # Load LiDAR ground truth
        lidar_gt_file = os.path.join(self.agri_slam_dir, "groundtruth_lidar.csv")
        if os.path.exists(lidar_gt_file):
            self.lidar_gt_poses = self._load_poses_from_csv(lidar_gt_file)
            self.gt_poses = self.lidar_gt_poses
            self.gt_pose_provided = True
            print(f"Loaded {len(self.gt_poses)} LiDAR ground truth poses")
        
        # Load camera ground truth  
        cam_gt_file = os.path.join(self.agri_slam_dir, "groundtruth_cam_2.csv")
        if os.path.exists(cam_gt_file):
            self.cam_gt_poses = self._load_poses_from_csv(cam_gt_file)
            print(f"Loaded {len(self.cam_gt_poses)} camera ground truth poses")
        
        # Camera parameters
        self.left_cam_name = "cam2"
        self.main_cam_name = self.left_cam_name
        
        # Camera intrinsics for cam_2 (calibrated values)
        K_cam = np.array([
            [737.81004, 0.0, 975.08049],
            [0.0, 737.95291, 569.79751],
            [0.0, 0.0, 1.0]
        ])
        
        # Camera to LiDAR transformation (calibrated)
        T_cam_lidar = np.array([
            [-0.65657749, -0.75423956, -0.00535658, 0.08879001],
            [0.02179564, -0.02607131, 0.99942245, -0.40550301],
            [-0.75394360, 0.65608153, 0.03355697, -0.17441673],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        ])      
        
        # Store calibration in dictionaries expected by PINGS
        self.K_mats = {self.left_cam_name: K_cam}
        self.T_c_l_mats = {self.left_cam_name: T_cam_lidar}
        
        # Set image dimensions based on intrinsic parameters
        # cx and cy suggest image size around 1920x1200 (cx≈975, cy≈570)
        H, W = 1200, 1920  # Updated based on camera calibration
        self.cam_widths = {self.left_cam_name: W}
        self.cam_heights = {self.left_cam_name: H}
        
        # Create Open3D intrinsic for depth/mono processing
        import open3d as o3d
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            height=H,
            width=W,
            fx=K_cam[0,0],
            fy=K_cam[1,1],
            cx=K_cam[0,2],
            cy=K_cam[1,2]
        )
        self.intrinsics_o3d = {self.left_cam_name: intrinsic}
        
        self.deskew_off = False
        
        # Create timestamp mappings
        self._create_timestamp_mappings()
        
    def _load_poses_from_csv(self, csv_file):
        """Load poses from CSV file with timestamp,tx,ty,tz,qx,qy,qz,qw format"""
        poses = []
        timestamps = []
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = row['timestamp']
                tx, ty, tz = float(row['tx']), float(row['ty']), float(row['tz'])
                qx, qy, qz, qw = float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])
                
                # Convert quaternion to rotation matrix
                quat = [qx, qy, qz, qw]
                rot_matrix = R.from_quat(quat).as_matrix()
                
                # Create 4x4 transformation matrix
                pose = np.eye(4)
                pose[:3, :3] = rot_matrix
                pose[:3, 3] = [tx, ty, tz]
                
                poses.append(pose)
                timestamps.append(timestamp)
                
        return np.array(poses)
    
    def _create_timestamp_mappings(self):
        """Create mappings between LiDAR and camera timestamps"""
        # Extract timestamps from filenames
        self.lidar_timestamps = []
        for scan_file in self.scan_files:
            filename = os.path.basename(scan_file)
            timestamp = filename.replace('.ply', '')
            self.lidar_timestamps.append(timestamp)
            
        self.img_timestamps = []
        for img_file in self.img_files:
            filename = os.path.basename(img_file)
            timestamp = filename.replace('.jpg', '')
            self.img_timestamps.append(timestamp)
    
    def __getitem__(self, idx):
        if idx >= len(self.scan_files):
            raise IndexError(f"Index {idx} out of range for {len(self.scan_files)} scans")
            
        # Load point cloud
        points = self.scans(idx)
        point_ts = self.get_timestamps(points, self.scan_files[idx])
        
        # Add point_lidar_idx like IPBCar (all zeros for single lidar)
        point_lidar_idx = np.zeros_like(point_ts)
        
        frame_data = {}
        frame_data["point_ts"] = point_ts
        
        if self.load_img and self.image_available:
            # Find closest image timestamp to LiDAR timestamp
            lidar_ts = self.lidar_timestamps[idx]
            closest_img_idx = self._find_closest_image(lidar_ts)
            
            if closest_img_idx is not None:
                img = self.read_img(self.img_files[closest_img_idx])
                
                # Apply sky removal if enabled
                if self.use_sky_removal and hasattr(self, 'mask_paths'):
                    mask_path = self.mask_paths[closest_img_idx]
                    img = self.apply_sky_mask(img, mask_path)

                # Create RGB colors for points (start with -1 like IPBCar)
                points_rgb = -1.0 * np.ones_like(points)  # N,4, last channel for the mask
                
                # Project points to camera to get colors
                points_rgb, depth_map = self.project_points_to_cam(
                    points, 
                    points_rgb, 
                    img, 
                    self.T_c_l_mats[self.main_cam_name], 
                    self.K_mats[self.main_cam_name]
                )
                
                # Filter points that got colors from projection (optional)
                if self.use_only_colorized_points:
                    with_rgb_mask = (points_rgb[:, 3] == 0)  # points_rgb[:, 3] is the color mask
                    points = points[with_rgb_mask]
                    points_rgb = points_rgb[with_rgb_mask]
                    point_ts = point_ts[with_rgb_mask]
                    point_lidar_idx = point_lidar_idx[with_rgb_mask]
                
                # Combine xyz with rgb like IPBCar: [x, y, z, r, g, b]
                points = np.hstack((points[:, :3], points_rgb[:, :3]))
                
                # Add image data (skip depth like IPBCar)
                img_dict = {self.main_cam_name: img}
                frame_data["img"] = img_dict
            else:
                # No matching image found, keep original points format
                pass
        else:
            # No images available, just return points as N,4 [x,y,z,1]
            pass
            
        frame_data.update({"points": points, "point_ts": point_ts, "point_lidar_idx": point_lidar_idx})
        
        return frame_data
    
    def _find_closest_image(self, lidar_timestamp):
        """Find the closest image timestamp to the given LiDAR timestamp"""
        if not self.img_timestamps:
            return None
            
        # Convert timestamps to integers for comparison
        try:
            lidar_ts_int = int(lidar_timestamp.replace('-', ''))
            img_ts_ints = [int(ts.replace('-', '')) for ts in self.img_timestamps]
            
            # Find closest
            differences = [abs(img_ts - lidar_ts_int) for img_ts in img_ts_ints]
            min_idx = differences.index(min(differences))
            
            # Return closest match (increased tolerance to 500ms)
            if differences[min_idx] < 500000000:  # 500ms in nanoseconds
                return min_idx
            else:
                # If no close match, return the closest anyway for testing
                return min_idx
        except:
            # Fallback: return middle image if parsing fails
            return len(self.img_timestamps) // 2
    
    def apply_sky_mask(self, image, mask_path):
        """Apply sky removal using depth_anything masks"""
        try:
            # Load depth mask
            depth_mask = np.array(Image.open(mask_path))
            
            # Create sky mask: pixels with depth == threshold (0) are sky
            sky_mask = depth_mask == 0
            
            # Apply mask to image (set sky regions to black)
            masked_image = image.copy()
            masked_image[sky_mask] = [0, 0, 0]  # Set sky pixels to black
            
            return masked_image
            
        except Exception as e:
            print(f"Warning: Failed to apply sky mask for {mask_path}: {e}")
            return image  # Return original image if masking fails

    def __len__(self):
        return len(self.scan_files)
    
    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])
    
    def read_point_cloud(self, scan_file: str):
        """Read PLY point cloud file"""
        pcd = o3d.io.read_point_cloud(scan_file)
        points = np.asarray(pcd.points)
        
        # Filter out NaN values (common in Ouster point clouds for invalid returns)
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]
        
        # Add homogeneous coordinate column (set to 1.0 for all points) to match IPBCar format
        if points.shape[1] == 3:
            homogeneous = np.ones((points.shape[0], 1))
            points = np.hstack([points, homogeneous])
            
        return points.astype(np.float64)  # N, 4 (x, y, z, 1)
    
    def read_img(self, img_file: str):
        """Read image file"""
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    @staticmethod
    def get_timestamps(points, scan_file_path=None):
        """Get timestamps for points - parse from filename like IPBCar"""
        n_points = points.shape[0]
        
        if scan_file_path is not None:
            # Extract timestamp from filename
            filename = os.path.basename(scan_file_path)
            timestamp_str = filename.replace('.ply', '')
            
            try:
                # Parse format: 1744202088-061251861 -> seconds.nanoseconds
                parts = timestamp_str.split('-')
                if len(parts) == 2:
                    seconds = int(parts[0])
                    nanoseconds = int(parts[1])
                    timestamp_sec = seconds + nanoseconds * 1e-9
                    # Return per-point timestamps (all same for static scan)
                    return np.full(n_points, timestamp_sec)
            except:
                pass
        
        # Fallback: uniform timestamp for static scans
        return np.zeros(n_points)
    
    def get_frames_timestamps(self) -> np.ndarray:
        """Get timestamps for each frame"""
        timestamps = []
        for i, ts_str in enumerate(self.lidar_timestamps):
            # Convert timestamp string to seconds
            try:
                # Format: 1744202088-061251861 -> seconds.nanoseconds
                parts = ts_str.split('-')
                seconds = int(parts[0])
                nanoseconds = int(parts[1])
                timestamp = seconds + nanoseconds * 1e-9
                timestamps.append(timestamp)
            except:
                timestamps.append(float(i))  # Fallback to frame index
                
        return np.array(timestamps).reshape(-1, 1)
    
    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        """Apply calibration transformation if needed"""
        # For now, return poses as-is since we're using LiDAR frame
        return poses
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        """Project 3D points to camera image"""
        # points as np.numpy (N,4)
        points_homo = points.copy()
        points_homo[:, 3] = 1  # homo coordinate
        
        # Transform LiDAR points to camera coordinate
        points_cam = np.matmul(T_c_l, points_homo.T).T  # N, 4
        points_cam = points_cam[:, :3]  # N, 3
        
        # Project to image space
        u, v, depth = self.perspective_cam2image(points_cam.T, K_mat)
        u = u.astype(np.int32)
        v = v.astype(np.int32)
        
        img_height, img_width, _ = img.shape
        
        # Prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width, 1))
        
        # Create mask for valid projections
        mask = np.logical_and(
            np.logical_and(
                np.logical_and(u >= 0, u < img_width),
                v >= 0
            ),
            v < img_height
        )
        
        # Distance filtering
        min_depth = 1.0
        max_depth = 100.0
        mask = np.logical_and(
            np.logical_and(mask, depth > min_depth),
            depth < max_depth
        )
        
        v_valid = v[mask]
        u_valid = u[mask]
        
        # Fill depth map
        depth_map[v_valid, u_valid, 0] = depth[mask]
        
        # Assign colors to points
        points_rgb[mask, :3] = img[v_valid, u_valid].astype(np.float64) / 255.0  # 0-1
        points_rgb[mask, 3] = 0  # has color flag
        
        return points_rgb, depth_map
        
    def perspective_cam2image(self, points, K_mat):
        """Project camera coordinate points to image"""
        ndim = points.ndim
        if ndim == 2:
            points = points[np.newaxis, :, :]
            
        points_proj = np.matmul(K_mat[:3, :3].reshape([1, 3, 3]), points)
        depth = points_proj[:, 2, :]
        depth[depth == 0] = -1e-6
        u = np.round(points_proj[:, 0, :] / np.abs(depth)).astype(int)
        v = np.round(points_proj[:, 1, :] / np.abs(depth)).astype(int)
        
        if ndim == 2:
            u = u[0]
            v = v[0]
            depth = depth[0]
            
        return u, v, depth

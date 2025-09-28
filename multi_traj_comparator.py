#!/usr/bin/env python3
"""
Multi-Camera Trajectory Comparison Tool

This script loads ground truth and predicted trajectories for multiple cameras,
aligns them individually using 6DoF rigid transformation, and computes ATE metrics
for each camera, providing both individual and average results.

Usage:
    python3 multi_traj_comparator.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

def rigid_alignment_6dof(P, Q):
    """
    Compute 6DoF rigid alignment (rotation + translation, no scale).
    
    Args:
        P (Nx3): Source points (predicted)
        Q (Nx3): Target points (ground truth)
    
    Returns:
        tuple: (R, t) - rotation matrix, translation vector
    """
    assert P.shape == Q.shape
    n, m = P.shape

    # Compute centroids
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    
    # Center the points
    P_centered = P - mu_P
    Q_centered = Q - mu_Q
    
    # Compute cross-covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD of cross-covariance
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = mu_Q - R @ mu_P
    
    return R, t

def umeyama_alignment_2d(P, Q):
    """
    Umeyama algorithm for 2D similarity transformation (rotation, translation, scale).
    
    Args:
        P (Nx2): Source points (predicted) in 2D
        Q (Nx2): Target points (ground truth) in 2D
    
    Returns:
        tuple: (R, t, s) - 2x2 rotation matrix, 2D translation vector, scale factor
    """
    assert P.shape == Q.shape
    assert P.shape[1] == 2, "Input must be 2D points"
    n, m = P.shape

    # Compute centroids
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    
    # Center the points
    P_centered = P - mu_P
    Q_centered = Q - mu_Q
    
    # Compute variances
    var_P = np.mean(np.sum(P_centered**2, axis=1))
    var_Q = np.mean(np.sum(Q_centered**2, axis=1))
    
    # Compute cross-covariance matrix
    H = P_centered.T @ Q_centered / n
    
    # SVD of cross-covariance
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1) for 2D
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    trace_TS = np.sum(S)
    if var_P > 1e-8:
        s = trace_TS / var_P
    else:
        s = 1.0
    
    # Compute translation
    t = mu_Q - s * R @ mu_P
    
    return R, t, s

def calculate_ate_metrics_2d(gt_positions_2d, pred_positions_2d):
    """
    Calculate Absolute Trajectory Error (ATE) metrics for 2D trajectories.
    
    Args:
        gt_positions_2d: Ground truth 2D positions (aligned)
        pred_positions_2d: Predicted 2D positions (aligned)
        
    Returns:
        dict: Dictionary with ATE metrics for 2D
    """
    # Use overlapping length
    n = min(len(gt_positions_2d), len(pred_positions_2d))
    errors = np.linalg.norm(gt_positions_2d[:n] - pred_positions_2d[:n], axis=1)
    
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'max': np.max(errors),
        'min': np.min(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'num_poses': n
    }

def umeyama_alignment(P, Q):
    """
    Umeyama algorithm for similarity transformation (rotation, translation, scale).
    
    Args:
        P (Nx3): Source points (predicted)
        Q (Nx3): Target points (ground truth)
    
    Returns:
        tuple: (R, t, s) - rotation matrix, translation vector, scale factor
    """
    assert P.shape == Q.shape
    n, m = P.shape

    # Compute centroids
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    
    # Center the points
    P_centered = P - mu_P
    Q_centered = Q - mu_Q
    
    # Compute variances
    var_P = np.mean(np.sum(P_centered**2, axis=1))
    var_Q = np.mean(np.sum(Q_centered**2, axis=1))
    
    # Compute cross-covariance matrix
    H = P_centered.T @ Q_centered / n
    
    # SVD of cross-covariance
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    trace_TS = np.sum(S)
    if var_P > 1e-8:
        s = trace_TS / var_P
    else:
        s = 1.0
    
    # Compute translation
    t = mu_Q - s * R @ mu_P
    
    return R, t, s

def align_trajectories_6dof_with_timestamps(gt_traj, pred_traj):
    """
    Align trajectories using 6DoF rigid alignment (no scale) and timestamp matching.
    Tests multiple initial transformations to find the best match.
    
    Args:
        gt_traj: Ground truth trajectory dictionary with timestamps
        pred_traj: Predicted trajectory dictionary with frame_ids
    Returns:
        tuple: (gt_aligned, pred_aligned) - optimally aligned trajectories
    """
    gt_positions = gt_traj['positions']
    pred_positions = pred_traj['positions']
    gt_timestamps = gt_traj['timestamps']
    pred_frame_ids = pred_traj['frame_ids']
    
    if len(gt_positions) == 0 or len(pred_positions) == 0:
        return gt_positions, pred_positions

    # Debug: Print coordinate ranges
    print(f"  GT position range: X[{gt_positions[:, 0].min():.3f}, {gt_positions[:, 0].max():.3f}], "
          f"Y[{gt_positions[:, 1].min():.3f}, {gt_positions[:, 1].max():.3f}], "
          f"Z[{gt_positions[:, 2].min():.3f}, {gt_positions[:, 2].max():.3f}]")
    print(f"  Pred position range: X[{pred_positions[:, 0].min():.3f}, {pred_positions[:, 0].max():.3f}], "
          f"Y[{pred_positions[:, 1].min():.3f}, {pred_positions[:, 1].max():.3f}], "
          f"Z[{pred_positions[:, 2].min():.3f}, {pred_positions[:, 2].max():.3f}]")

    # Enhanced timestamp-based matching with precise temporal alignment
    print(f"  GT timestamps range: [{gt_timestamps.min():.3f}, {gt_timestamps.max():.3f}] sec")
    print(f"  Pred frame_ids range: [{pred_frame_ids.min()}, {pred_frame_ids.max()}]")
    
    # Convert frame_ids to timestamps (assuming constant frame rate)
    # Estimate frame rate from predicted trajectory length and ground truth time span
    gt_duration = gt_timestamps.max() - gt_timestamps.min()
    pred_frame_span = pred_frame_ids.max() - pred_frame_ids.min()
    
    if pred_frame_span > 0:
        estimated_fps = pred_frame_span / gt_duration
        pred_timestamps = gt_timestamps.min() + (pred_frame_ids - pred_frame_ids.min()) / estimated_fps
        print(f"  Estimated frame rate: {estimated_fps:.2f} fps")
        print(f"  Converted pred timestamps range: [{pred_timestamps.min():.3f}, {pred_timestamps.max():.3f}] sec")
    else:
        # Fallback: distribute predicted frames evenly across GT time range
        pred_timestamps = np.linspace(gt_timestamps.min(), gt_timestamps.max(), len(pred_frame_ids))
        print(f"  Using uniform timestamp distribution for predicted trajectory")
    
    # Find overlapping time range
    min_time = max(gt_timestamps.min(), pred_timestamps.min())
    max_time = min(gt_timestamps.max(), pred_timestamps.max())
    print(f"  Temporal overlap: [{min_time:.3f}, {max_time:.3f}] sec")
    
    # Filter both trajectories to overlapping time range
    gt_mask = (gt_timestamps >= min_time) & (gt_timestamps <= max_time)
    pred_mask = (pred_timestamps >= min_time) & (pred_timestamps <= max_time)
    
    gt_filtered = gt_positions[gt_mask]
    pred_filtered = pred_positions[pred_mask]
    gt_time_filtered = gt_timestamps[gt_mask]
    pred_time_filtered = pred_timestamps[pred_mask]
    
    print(f"  GT poses in overlap: {len(gt_filtered)}")
    print(f"  Pred poses in overlap: {len(pred_filtered)}")
    
    if len(gt_filtered) < 3 or len(pred_filtered) < 3:
        print("  Warning: Not enough overlapping poses, using sequential matching")
        n = min(len(gt_positions), len(pred_positions))
        gt_subset = gt_positions[:n]
        pred_subset = pred_positions[:n]
    else:
        # Create synchronized trajectory using interpolation
        # Use the timeline with more poses as the reference
        if len(gt_filtered) >= len(pred_filtered):
            # Use GT timeline as reference
            reference_times = gt_time_filtered
            gt_subset = gt_filtered
            
            # Interpolate predicted trajectory to GT timestamps
            if len(pred_filtered) > 1:
                from scipy.interpolate import interp1d
                pred_interp_x = interp1d(pred_time_filtered, pred_filtered[:, 0], 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                pred_interp_y = interp1d(pred_time_filtered, pred_filtered[:, 1], 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                pred_interp_z = interp1d(pred_time_filtered, pred_filtered[:, 2], 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                
                pred_subset = np.column_stack([
                    pred_interp_x(reference_times),
                    pred_interp_y(reference_times),
                    pred_interp_z(reference_times)
                ])
            else:
                pred_subset = pred_filtered
                gt_subset = gt_filtered[:len(pred_subset)]
        else:
            # Use predicted timeline as reference
            reference_times = pred_time_filtered
            pred_subset = pred_filtered
            
            # Interpolate GT trajectory to predicted timestamps
            if len(gt_filtered) > 1:
                from scipy.interpolate import interp1d
                gt_interp_x = interp1d(gt_time_filtered, gt_filtered[:, 0], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                gt_interp_y = interp1d(gt_time_filtered, gt_filtered[:, 1], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                gt_interp_z = interp1d(gt_time_filtered, gt_filtered[:, 2], 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                
                gt_subset = np.column_stack([
                    gt_interp_x(reference_times),
                    gt_interp_y(reference_times),
                    gt_interp_z(reference_times)
                ])
            else:
                gt_subset = gt_filtered
                pred_subset = pred_filtered[:len(gt_subset)]
        
        print(f"  Timestamp-synchronized poses: {len(gt_subset)}")
        
        # Calculate temporal alignment quality
        if len(reference_times) > 1:
            time_diffs = np.diff(reference_times)
            avg_time_interval = np.mean(time_diffs)
            print(f"  Average time interval: {avg_time_interval:.4f} sec")
            print(f"  Temporal resolution: {1/avg_time_interval:.1f} Hz")

    # Calculate initial direction for enhanced alignment
    initial_direction_rotation = None
    if len(gt_subset) >= 5 and len(pred_subset) >= 5:
        # Use a larger window (first 5 poses) to get more reliable direction vectors
        direction_window = min(5, len(gt_subset), len(pred_subset))
        
        # Calculate direction vectors using first and last pose in window
        gt_dir_xy = gt_subset[direction_window-1, :2] - gt_subset[0, :2]
        pred_dir_xy = pred_subset[direction_window-1, :2] - pred_subset[0, :2]
        
        # Normalize direction vectors
        gt_dir_norm = np.linalg.norm(gt_dir_xy)
        pred_dir_norm = np.linalg.norm(pred_dir_xy)
        
        print(f"  Calculating initial direction from first {direction_window} poses")
        print(f"  GT direction magnitude: {gt_dir_norm:.6f}, Pred direction magnitude: {pred_dir_norm:.6f}")
        
        if gt_dir_norm > 1e-4 and pred_dir_norm > 1e-4:
            gt_unit_xy = gt_dir_xy / gt_dir_norm
            pred_unit_xy = pred_dir_xy / pred_dir_norm
            
            # Calculate angle between directions in XY plane
            cos_angle = np.clip(np.dot(gt_unit_xy, pred_unit_xy), -1, 1)
            sin_angle = np.cross(gt_unit_xy, pred_unit_xy)
            initial_angle_rad = np.arctan2(sin_angle, cos_angle)
            initial_angle_deg = np.rad2deg(initial_angle_rad)
            
            print(f"  Initial XY direction misalignment: {initial_angle_deg:.2f}°")
            
            # Store the initial direction rotation for integration with coarse transforms
            initial_direction_rotation = np.array([
                [np.cos(initial_angle_rad), -np.sin(initial_angle_rad), 0],
                [np.sin(initial_angle_rad),  np.cos(initial_angle_rad), 0],
                [0, 0, 1]
            ])

    # Focus on XY plane alignment with fine-grained rotation testing
    print("  Using pure 6DoF alignment with XY plane focus (no scaling)")

    # Enhanced coarse transformations including initial direction alignment
    coarse_transforms = [
        # Identity
        (np.eye(3), "Identity"),
        
        # 180-degree rotations around each axis
        (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), "180° around X"),
        (np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]), "180° around Y"),
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]), "180° around Z"),
        (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), "180° around X+Y"),
    ]
    
    # Add initial direction alignment to each coarse transform if available
    if initial_direction_rotation is not None:
        enhanced_transforms = []
        for base_R, base_desc in coarse_transforms:
            # Add base transform
            enhanced_transforms.append((base_R, base_desc))
            # Add base transform + initial direction alignment
            combined_R = initial_direction_rotation @ base_R
            enhanced_transforms.append((combined_R, f"{base_desc} + Initial Direction"))
        coarse_transforms = enhanced_transforms
    
    # Find best coarse alignment
    best_coarse_error = float('inf')
    best_coarse_transform = None
    best_coarse_description = ""
    
    print(f"  Testing {len(coarse_transforms)} coarse transformations...")
    
    for initial_R, description in coarse_transforms:
        try:
            pred_transformed = (initial_R @ pred_subset.T).T
            R, t = rigid_alignment_6dof(pred_transformed, gt_subset)
            pred_aligned = (R @ (initial_R @ pred_subset.T)).T + t
            errors = np.linalg.norm(gt_subset - pred_aligned, axis=1)
            mean_error = np.mean(errors)
            
            # Special logging for direction-enhanced transforms
            if "Initial Direction" in description:
                print(f"    {description}: {mean_error:.4f} m")
            
            if mean_error < best_coarse_error:
                best_coarse_error = mean_error
                best_coarse_transform = initial_R
                best_coarse_description = description
        except:
            continue
    
    print(f"  Best coarse alignment: {best_coarse_description}")
    
    # If we have initial direction information and it wasn't used in the best coarse transform,
    # prioritize direction-aware fine-tuning
    use_direction_priority = (initial_direction_rotation is not None and 
                            "Initial Direction" not in best_coarse_description)
    
    if use_direction_priority:
        print(f"  Applying direction-aware fine-tuning (initial misalignment: {np.rad2deg(initial_angle_rad):.1f}°)")
    
    # Now fine-tune with XY plane rotations around the best coarse transform
    fine_rotation_angles = np.linspace(-10, 10, 41)  # -10° to +10° in 0.5° increments
    
    best_error = float('inf')
    best_alignment = None
    best_description = ""
    best_xy_angle = 0
    
    for angle_deg in fine_rotation_angles:
        try:
            angle_rad = np.deg2rad(angle_deg)
            
            # Create XY plane rotation matrix (rotation around Z axis)
            xy_rotation = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad),  np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            
            # Combine coarse transform with fine XY rotation
            combined_transform = xy_rotation @ best_coarse_transform
            
            # Apply combined transformation
            pred_transformed = (combined_transform @ pred_subset.T).T
            
            # Apply 6DoF rigid alignment
            R, t = rigid_alignment_6dof(pred_transformed, gt_subset)
            
            # Apply full transformation
            pred_aligned = (R @ (combined_transform @ pred_subset.T)).T + t
            
            # Calculate XY-focused error (weight XY more than Z)
            errors_3d = np.linalg.norm(gt_subset - pred_aligned, axis=1)
            errors_xy = np.linalg.norm(gt_subset[:, :2] - pred_aligned[:, :2], axis=1)
            
            # Combined error: 80% XY plane, 20% full 3D
            combined_error = 0.8 * np.mean(errors_xy) + 0.2 * np.mean(errors_3d)
            
            # Enhanced trajectory shape analysis for XY plane
            if len(gt_subset) > 2:
                # XY trajectory directions
                gt_dirs_xy = np.diff(gt_subset[:, :2], axis=0)
                pred_dirs_xy = np.diff(pred_aligned[:, :2], axis=0)
                
                # Normalize XY directions
                gt_norms_xy = np.linalg.norm(gt_dirs_xy, axis=1)
                pred_norms_xy = np.linalg.norm(pred_dirs_xy, axis=1)
                
                valid_xy = (gt_norms_xy > 1e-6) & (pred_norms_xy > 1e-6)
                if np.any(valid_xy):
                    gt_unit_xy = gt_dirs_xy[valid_xy] / gt_norms_xy[valid_xy, np.newaxis]
                    pred_unit_xy = pred_dirs_xy[valid_xy] / pred_norms_xy[valid_xy, np.newaxis]
                    
                    # XY direction cosine similarity
                    cos_sim_xy = np.sum(gt_unit_xy * pred_unit_xy, axis=1)
                    direction_score_xy = np.mean(np.clip(cos_sim_xy, -1, 1))
                    
                    # XY curvature similarity
                    if len(gt_subset) > 3:
                        gt_curvature_xy = np.diff(gt_dirs_xy, axis=0)
                        pred_curvature_xy = np.diff(pred_dirs_xy, axis=0)
                        
                        gt_curv_norms_xy = np.linalg.norm(gt_curvature_xy, axis=1)
                        pred_curv_norms_xy = np.linalg.norm(pred_curvature_xy, axis=1)
                        
                        curv_valid_xy = (gt_curv_norms_xy > 1e-6) & (pred_curv_norms_xy > 1e-6)
                        if np.any(curv_valid_xy):
                            gt_curv_unit_xy = gt_curvature_xy[curv_valid_xy] / gt_curv_norms_xy[curv_valid_xy, np.newaxis]
                            pred_curv_unit_xy = pred_curvature_xy[curv_valid_xy] / pred_curv_norms_xy[curv_valid_xy, np.newaxis]
                            
                            curv_cos_sim_xy = np.sum(gt_curv_unit_xy * pred_curv_unit_xy, axis=1)
                            curvature_score_xy = np.mean(np.clip(curv_cos_sim_xy, -1, 1))
                        else:
                            curvature_score_xy = 0.0
                    else:
                        curvature_score_xy = 0.0
                    
                    # XY-focused combined score: 40% position, 35% direction, 25% curvature
                    total_error = (0.4 * combined_error + 
                                 0.35 * (1.0 - direction_score_xy) * np.mean(gt_norms_xy) + 
                                 0.25 * (1.0 - curvature_score_xy) * np.mean(gt_norms_xy))
                else:
                    total_error = combined_error
            else:
                total_error = combined_error
            
            if total_error < best_error:
                best_error = total_error
                best_alignment = (combined_transform, R, t)
                best_description = f"{best_coarse_description} + {angle_deg:.1f}° XY rotation"
                best_xy_angle = angle_deg
                
        except Exception as e:
            continue
    
    print(f"  Fine XY rotation: {best_xy_angle:.1f}°")
    
    if best_alignment is None:
        print("  Warning: No valid 6DoF XY alignment found, using identity transform")
        return gt_positions, pred_positions
    
    # Apply best transformation to full predicted trajectory with timestamp alignment
    best_combined_R, best_R, best_t = best_alignment
    
    # For the final result, we need to align the full trajectories based on timestamps
    # Convert full predicted frame_ids to timestamps using the same method
    if pred_frame_span > 0:
        pred_timestamps_full = gt_timestamps.min() + (pred_frame_ids - pred_frame_ids.min()) / estimated_fps
    else:
        pred_timestamps_full = np.linspace(gt_timestamps.min(), gt_timestamps.max(), len(pred_frame_ids))
    
    # Find the temporal overlap for full trajectories
    min_time_full = max(gt_timestamps.min(), pred_timestamps_full.min())
    max_time_full = min(gt_timestamps.max(), pred_timestamps_full.max())
    
    # Filter full trajectories to overlapping time range
    gt_mask_full = (gt_timestamps >= min_time_full) & (gt_timestamps <= max_time_full)
    pred_mask_full = (pred_timestamps_full >= min_time_full) & (pred_timestamps_full <= max_time_full)
    
    gt_filtered_full = gt_positions[gt_mask_full]
    pred_filtered_full = pred_positions[pred_mask_full]
    gt_time_filtered_full = gt_timestamps[gt_mask_full]
    pred_time_filtered_full = pred_timestamps_full[pred_mask_full]
    
    # Apply transformation to the filtered predicted trajectory
    pred_transformed_filtered = (best_combined_R @ pred_filtered_full.T).T
    pred_aligned_filtered = (best_R @ pred_transformed_filtered.T).T + best_t
    
    # Create synchronized final trajectories using interpolation
    if len(gt_filtered_full) >= len(pred_filtered_full):
        # Use GT timeline as reference for final output
        final_gt = gt_filtered_full
        
        # Interpolate aligned predicted trajectory to GT timestamps
        if len(pred_filtered_full) > 1:
            pred_interp_x = interp1d(pred_time_filtered_full, pred_aligned_filtered[:, 0], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            pred_interp_y = interp1d(pred_time_filtered_full, pred_aligned_filtered[:, 1], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            pred_interp_z = interp1d(pred_time_filtered_full, pred_aligned_filtered[:, 2], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            
            final_pred = np.column_stack([
                pred_interp_x(gt_time_filtered_full),
                pred_interp_y(gt_time_filtered_full),
                pred_interp_z(gt_time_filtered_full)
            ])
        else:
            final_pred = pred_aligned_filtered
            final_gt = gt_filtered_full[:len(final_pred)]
    else:
        # Use predicted timeline as reference for final output
        final_pred = pred_aligned_filtered
        
        # Interpolate GT trajectory to predicted timestamps
        if len(gt_filtered_full) > 1:
            gt_interp_x = interp1d(gt_time_filtered_full, gt_filtered_full[:, 0], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            gt_interp_y = interp1d(gt_time_filtered_full, gt_filtered_full[:, 1], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            gt_interp_z = interp1d(gt_time_filtered_full, gt_filtered_full[:, 2], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            
            final_gt = np.column_stack([
                gt_interp_x(pred_time_filtered_full),
                gt_interp_y(pred_time_filtered_full),
                gt_interp_z(pred_time_filtered_full)
            ])
        else:
            final_gt = gt_filtered_full
            final_pred = pred_aligned_filtered[:len(final_gt)]
    
    print(f"  Final synchronized trajectory length: {len(final_gt)} poses")
    print(f"  Best 6DoF XY alignment: {best_description}")
    print(f"  Alignment error: {best_error:.4f} m")
    
    return final_gt, final_pred

def parse_csv_trajectory(file_path):
    """
    Parse CSV format trajectory file with timestamp, tx, ty, tz, qx, qy, qz, qw.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        dict: Dictionary with timestamps, positions, and quaternions
    """
    df = pd.read_csv(file_path)
    
    # Convert timestamp string to numeric
    timestamps = []
    for ts in df['timestamp']:
        timestamp_sec = float(ts.split('-')[0])
        timestamps.append(timestamp_sec)
    
    timestamps = np.array(timestamps)
    positions = df[['tx', 'ty', 'tz']].values
    quaternions = df[['qx', 'qy', 'qz', 'qw']].values
    
    return {
        'timestamps': timestamps,
        'positions': positions,
        'quaternions': quaternions
    }

def parse_tum_trajectory(file_path):
    """
    Parse TUM format trajectory file with transformation matrices.
    
    Args:
        file_path (str): Path to TUM format file
        
    Returns:
        dict: Dictionary with frame_ids, positions, and quaternions
    """
    positions = []
    quaternions = []
    frame_ids = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
                
            parts = line.split()
            if len(parts) != 13:
                continue
                
            frame_id = int(float(parts[0]))
            # Transformation matrix elements (3x4 matrix)
            # Format: frame_id T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23
            T = np.array([
                [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])],
                [float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12])],
                [0, 0, 0, 1]
            ])
            
            # Invert the transformation matrix to get the inverse pose
            T_inv = T #np.linalg.inv(T)
            
            # Extract position (translation) from inverted matrix
            position = T_inv[:3, 3]
            
            # Extract rotation matrix from inverted matrix and convert to quaternion
            rotation_matrix = T_inv[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()  # [x, y, z, w]
            
            frame_ids.append(frame_id)
            positions.append(position)
            quaternions.append(quaternion)
    
    return {
        'frame_ids': np.array(frame_ids),
        'positions': np.array(positions),
        'quaternions': np.array(quaternions)
    }

def save_aligned_trajectories_xy(gt_aligned, pred_aligned, gt_timestamps, camera_name, output_dir='.'):
    """
    Save the aligned trajectories in CSV format with timestamps, focusing on XY plane.
    
    Args:
        gt_aligned: Aligned ground truth positions
        pred_aligned: Aligned predicted positions
        gt_timestamps: Ground truth timestamps
        camera_name: Camera identifier
        output_dir: Output directory for saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have the same number of poses and timestamps
    n = min(len(gt_aligned), len(pred_aligned), len(gt_timestamps))
    
    # Create DataFrames for GT and predicted trajectories
    gt_df = pd.DataFrame({
        'timestamp': gt_timestamps[:n],
        'tx': gt_aligned[:n, 0],  # X coordinate
        'ty': gt_aligned[:n, 1],  # Y coordinate
        'tz': gt_aligned[:n, 2],  # Z coordinate (kept for completeness)
        'qx': 0.0,  # Placeholder quaternion values
        'qy': 0.0,
        'qz': 0.0,
        'qw': 1.0
    })
    
    pred_df = pd.DataFrame({
        'timestamp': gt_timestamps[:n],  # Using GT timestamps for alignment
        'tx': pred_aligned[:n, 0],  # X coordinate
        'ty': pred_aligned[:n, 1],  # Y coordinate
        'tz': pred_aligned[:n, 2],  # Z coordinate (kept for completeness)
        'qx': 0.0,  # Placeholder quaternion values
        'qy': 0.0,
        'qz': 0.0,
        'qw': 1.0
    })
    
    # Save files
    gt_filename = os.path.join(output_dir, f'gt_cam_{camera_name}_aligned.csv')
    pred_filename = os.path.join(output_dir, f'pred_cam_{camera_name}_aligned.csv')
    
    gt_df.to_csv(gt_filename, index=False)
    pred_df.to_csv(pred_filename, index=False)
    
    # Also save XY-only versions for specific XY plane analysis
    gt_xy_df = pd.DataFrame({
        'timestamp': gt_timestamps[:n],
        'x': gt_aligned[:n, 0],
        'y': gt_aligned[:n, 1]
    })
    
    pred_xy_df = pd.DataFrame({
        'timestamp': gt_timestamps[:n],
        'x': pred_aligned[:n, 0],
        'y': pred_aligned[:n, 1]
    })
    
    gt_xy_filename = os.path.join(output_dir, f'gt_cam_{camera_name}_xy_aligned.csv')
    pred_xy_filename = os.path.join(output_dir, f'pred_cam_{camera_name}_xy_aligned.csv')
    
    gt_xy_df.to_csv(gt_xy_filename, index=False)
    pred_xy_df.to_csv(pred_xy_filename, index=False)
    
    print(f"  ✓ Saved aligned trajectories:")
    print(f"    - Full 3D GT: {gt_filename}")
    print(f"    - Full 3D Pred: {pred_filename}")
    print(f"    - XY-only GT: {gt_xy_filename}")
    print(f"    - XY-only Pred: {pred_xy_filename}")
    
    return {
        'gt_full': gt_filename,
        'pred_full': pred_filename,
        'gt_xy': gt_xy_filename,
        'pred_xy': pred_xy_filename
    }

def calculate_ate_metrics(gt_positions, pred_positions):
    """
    Calculate Absolute Trajectory Error (ATE) metrics.
    
    Args:
        gt_positions: Ground truth positions (aligned)
        pred_positions: Predicted positions (aligned)
        
    Returns:
        dict: Dictionary with ATE metrics
    """
    # Use overlapping length
    n = min(len(gt_positions), len(pred_positions))
    errors = np.linalg.norm(gt_positions[:n] - pred_positions[:n], axis=1)
    
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'max': np.max(errors),
        'min': np.min(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'num_poses': n
    }

def compare_single_trajectory(gt_traj, pred_traj, camera_name):
    """
    Compare a single camera's trajectory.
    
    Args:
        gt_traj: Ground truth trajectory dictionary
        pred_traj: Predicted trajectory dictionary
        camera_name: Name of the camera
        
    Returns:
        dict: ATE metrics for this camera
    """
    print(f"\n=== CAMERA {camera_name} ===")
    
    gt_positions = gt_traj['positions']
    pred_positions = pred_traj['positions']
    
    print(f"Ground truth trajectory: {len(gt_positions)} poses")
    print(f"Predicted trajectory: {len(pred_positions)} poses")
    
    # Align trajectories using 6DoF alignment with timestamp matching
    gt_aligned, pred_aligned = align_trajectories_6dof_with_timestamps(gt_traj, pred_traj)
    
    # Extract XY projections from aligned trajectories
    print(f"\n--- XY PLANE ANALYSIS ---")
    gt_xy = gt_aligned[:, :2]  # Extract X,Y coordinates only
    pred_xy = pred_aligned[:, :2]  # Extract X,Y coordinates only
    
    print(f"Extracted XY trajectories: {len(gt_xy)} poses")
    print(f"GT XY range: X[{gt_xy[:, 0].min():.3f}, {gt_xy[:, 0].max():.3f}], Y[{gt_xy[:, 1].min():.3f}, {gt_xy[:, 1].max():.3f}]")
    print(f"Pred XY range: X[{pred_xy[:, 0].min():.3f}, {pred_xy[:, 0].max():.3f}], Y[{pred_xy[:, 1].min():.3f}, {pred_xy[:, 1].max():.3f}]")
    
    # Apply Umeyama alignment on XY plane trajectories with first pose as anchor
    print("Applying Umeyama alignment on XY plane trajectories with first pose anchoring...")
    
    print(f"  First pose anchoring: GT start at [{gt_xy[0, 0]:.4f}, {gt_xy[0, 1]:.4f}]")
    print(f"  Applying Umeyama with first pose as fixed anchor point")
    
    # Apply standard Umeyama alignment first
    if len(gt_xy) > 1:
        R_xy_base, t_xy_base, s_xy_base = umeyama_alignment_2d(pred_xy, gt_xy)
    else:
        # Fallback for single pose
        R_xy_base = np.eye(2)
        t_xy_base = np.zeros(2)
        s_xy_base = 1.0
    
    # Apply base Umeyama transformation
    pred_xy_umeyama_base = s_xy_base * (R_xy_base @ pred_xy.T).T + t_xy_base
    
    # Calculate the offset between transformed first pose and GT first pose
    first_pose_offset = gt_xy[0] - pred_xy_umeyama_base[0]
    
    # Apply the first pose correction to the entire trajectory
    pred_xy_umeyama_corrected = pred_xy_umeyama_base + first_pose_offset
    
    # Test fine rotation adjustments around the first pose anchor
    print("  Testing fine clockwise/counterclockwise rotations around first pose anchor...")
    
    # Test rotation angles from -10° to +10° in 0.5° increments
    test_angles = np.linspace(-10, 10, 41)  # 41 angles from -10° to +10°
    best_error = float('inf')
    best_angle = 0
    best_transformation = None
    
    for angle_deg in test_angles:
        angle_rad = np.deg2rad(angle_deg)
        
        # Create fine rotation matrix around the first pose (anchor point)
        fine_rotation = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        
        # Rotate around the first pose as anchor point
        # 1. Translate to make first pose the origin
        pred_xy_centered = pred_xy_umeyama_corrected - gt_xy[0]
        
        # 2. Apply rotation around origin (which is now the first pose)
        pred_xy_rotated = (fine_rotation @ pred_xy_centered.T).T
        
        # 3. Translate back to original position
        pred_xy_test = pred_xy_rotated + gt_xy[0]
        
        # The first pose should remain exactly at the anchor point
        pred_xy_test[0] = gt_xy[0]
        
        # Calculate alignment error
        errors = np.linalg.norm(gt_xy - pred_xy_test, axis=1)
        mean_error = np.mean(errors)
        
        if mean_error < best_error:
            best_error = mean_error
            best_angle = angle_deg
            best_transformation = (fine_rotation @ R_xy_base, t_xy_base, s_xy_base, pred_xy_test)
    
    # Use the best transformation
    R_xy, t_xy, s_xy, pred_xy_umeyama = best_transformation
    
    print(f"  Optimal fine rotation around anchor: {best_angle:.1f}° ({'clockwise' if best_angle < 0 else 'counterclockwise'})")
    print(f"  Best alignment error: {best_error:.4f} m")
    
    print(f"Umeyama 2D transformation (anchored to first pose):")
    print(f"  Rotation angle: {np.rad2deg(np.arctan2(R_xy[1,0], R_xy[0,0])):.2f}°")
    print(f"  Translation: [{t_xy[0]:.4f}, {t_xy[1]:.4f}]")
    print(f"  Scale factor: {s_xy:.6f}")
    print(f"  First pose anchor: [{gt_xy[0, 0]:.4f}, {gt_xy[0, 1]:.4f}]")
    
    # Verify starting point alignment
    start_error = np.linalg.norm(gt_xy[0] - pred_xy_umeyama[0])
    print(f"  Starting point alignment error: {start_error:.8f} m")
    
    # Calculate ATE metrics for original 3D alignment
    ate_metrics = calculate_ate_metrics(gt_aligned, pred_aligned)
    
    # Calculate ATE metrics for XY Umeyama alignment
    ate_metrics_xy = calculate_ate_metrics_2d(gt_xy, pred_xy_umeyama)
    
    print(f"\n3D ATE Results (6DoF rigid alignment):")
    print(f"  Mean error: {ate_metrics['mean']:.4f} m")
    print(f"  Std dev: {ate_metrics['std']:.4f} m") 
    print(f"  Median error: {ate_metrics['median']:.4f} m")
    print(f"  Max error: {ate_metrics['max']:.4f} m")
    print(f"  Min error: {ate_metrics['min']:.4f} m")
    print(f"  RMSE: {ate_metrics['rmse']:.4f} m")
    print(f"  Number of aligned poses: {ate_metrics['num_poses']}")
    
    print(f"\n2D XY ATE Results (Umeyama alignment):")
    print(f"  Mean error: {ate_metrics_xy['mean']:.4f} m")
    print(f"  Std dev: {ate_metrics_xy['std']:.4f} m") 
    print(f"  Median error: {ate_metrics_xy['median']:.4f} m")
    print(f"  Max error: {ate_metrics_xy['max']:.4f} m")
    print(f"  Min error: {ate_metrics_xy['min']:.4f} m")
    print(f"  RMSE: {ate_metrics_xy['rmse']:.4f} m")
    print(f"  Number of aligned poses: {ate_metrics_xy['num_poses']}")
    
    # Return both 3D and 2D metrics, plus the XY aligned trajectories
    return ate_metrics, ate_metrics_xy, gt_aligned, pred_aligned, gt_xy, pred_xy_umeyama

def plot_camera_trajectory_enhanced(gt_aligned, pred_aligned, gt_xy, pred_xy_umeyama, 
                                   ate_metrics_3d, ate_metrics_2d, camera_name, save_plots=False):
    """
    Enhanced plot showing both 3D and 2D XY trajectory comparisons for a single camera.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Camera {camera_name} Trajectory Comparison: 3D vs 2D XY Umeyama', fontsize=16)
    
    # 3D XY plot (from 6DoF alignment)
    ax1 = axes[0, 0]
    ax1.plot(gt_aligned[:, 0], gt_aligned[:, 1], 'b-', linewidth=2, label='Ground Truth')
    ax1.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'r-', linewidth=2, label='Predicted (6DoF)')
    ax1.scatter(gt_aligned[0, 0], gt_aligned[0, 1], c='blue', s=100, marker='o', label='GT Start')
    ax1.scatter(pred_aligned[0, 0], pred_aligned[0, 1], c='red', s=100, marker='o', label='Pred Start')
    ax1.scatter(gt_aligned[-1, 0], gt_aligned[-1, 1], c='blue', s=80, marker='s', label='GT End')
    ax1.scatter(pred_aligned[-1, 0], pred_aligned[-1, 1], c='red', s=80, marker='s', label='Pred End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('3D Trajectory XY View (6DoF Rigid)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # 2D XY plot (from Umeyama alignment)
    ax2 = axes[0, 1]
    ax2.plot(gt_xy[:, 0], gt_xy[:, 1], 'b-', linewidth=2, label='Ground Truth XY')
    ax2.plot(pred_xy_umeyama[:, 0], pred_xy_umeyama[:, 1], 'g-', linewidth=2, label='Predicted (Umeyama 2D)')
    ax2.scatter(gt_xy[0, 0], gt_xy[0, 1], c='blue', s=100, marker='o', label='GT Start')
    ax2.scatter(pred_xy_umeyama[0, 0], pred_xy_umeyama[0, 1], c='green', s=100, marker='o', label='Pred Start')
    ax2.scatter(gt_xy[-1, 0], gt_xy[-1, 1], c='blue', s=80, marker='s', label='GT End')
    ax2.scatter(pred_xy_umeyama[-1, 0], pred_xy_umeyama[-1, 1], c='green', s=80, marker='s', label='Pred End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('2D XY Trajectory (Umeyama)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Overlay comparison
    ax3 = axes[0, 2]
    ax3.plot(gt_aligned[:, 0], gt_aligned[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax3.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'r--', linewidth=2, label='6DoF Rigid', alpha=0.8)
    ax3.plot(pred_xy_umeyama[:, 0], pred_xy_umeyama[:, 1], 'g-', linewidth=2, label='Umeyama 2D', alpha=0.8)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Overlay Comparison')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # 3D Error plot
    n_3d = min(len(gt_aligned), len(pred_aligned))
    errors_3d = np.linalg.norm(gt_aligned[:n_3d] - pred_aligned[:n_3d], axis=1)
    
    ax4 = axes[1, 0]
    ax4.plot(errors_3d, 'red', linewidth=2, label='3D Error')
    ax4.axhline(errors_3d.mean(), color='orange', linestyle='--', linewidth=2, 
                label=f'Mean {errors_3d.mean():.3f} m')
    ax4.set_xlabel('Pose Index')
    ax4.set_ylabel('Error (m)')
    ax4.set_title('3D Point-wise Error (6DoF)')
    ax4.legend()
    ax4.grid(True)
    
    # 2D Error plot
    n_2d = min(len(gt_xy), len(pred_xy_umeyama))
    errors_2d = np.linalg.norm(gt_xy[:n_2d] - pred_xy_umeyama[:n_2d], axis=1)
    
    ax5 = axes[1, 1]
    ax5.plot(errors_2d, 'green', linewidth=2, label='2D XY Error')
    ax5.axhline(errors_2d.mean(), color='orange', linestyle='--', linewidth=2, 
                label=f'Mean {errors_2d.mean():.3f} m')
    ax5.set_xlabel('Pose Index')
    ax5.set_ylabel('Error (m)')
    ax5.set_title('2D XY Point-wise Error (Umeyama)')
    ax5.legend()
    ax5.grid(True)
    
    # Combined error histogram
    ax6 = axes[1, 2]
    ax6.hist(errors_3d, bins=20, alpha=0.7, color='red', label='3D Errors', density=True)
    ax6.hist(errors_2d, bins=20, alpha=0.7, color='green', label='2D XY Errors', density=True)
    ax6.axvline(errors_3d.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'3D Mean {errors_3d.mean():.3f} m')
    ax6.axvline(errors_2d.mean(), color='green', linestyle='--', linewidth=2, 
                label=f'2D Mean {errors_2d.mean():.3f} m')
    ax6.set_xlabel('Error (m)')
    ax6.set_ylabel('Density')
    ax6.set_title('Error Distribution Comparison')
    ax6.legend()
    ax6.grid(True)
    
    # Add metrics text box
    metrics_text = f"""3D Metrics (6DoF):
Mean: {ate_metrics_3d['mean']:.4f} m
RMSE: {ate_metrics_3d['rmse']:.4f} m
Std: {ate_metrics_3d['std']:.4f} m

2D XY Metrics (Umeyama):
Mean: {ate_metrics_2d['mean']:.4f} m
RMSE: {ate_metrics_2d['rmse']:.4f} m
Std: {ate_metrics_2d['std']:.4f} m"""
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'camera_{camera_name}_enhanced_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved enhanced plot: {filename}")
    
    plt.show()  # Show interactive plots

def plot_camera_trajectory(gt_aligned, pred_aligned, camera_name, save_plots=False):
    """
    Plot trajectory comparison for a single camera.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Camera {camera_name} Trajectory Comparison', fontsize=16)
    
    # XY plot
    ax1 = axes[0, 0]
    ax1.plot(gt_aligned[:, 0], gt_aligned[:, 1], 'b-', linewidth=2, label='Ground Truth')
    ax1.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'r-', linewidth=2, label='Predicted')
    ax1.scatter(gt_aligned[-1, 0], gt_aligned[-1, 1], c='blue', s=80, marker='s', label='GT End')
    ax1.scatter(pred_aligned[-1, 0], pred_aligned[-1, 1], c='red', s=80, marker='s', label='Pred End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # XZ plot
    ax2 = axes[0, 1]
    ax2.plot(gt_aligned[:, 0], gt_aligned[:, 2], 'b-', linewidth=2, label='Ground Truth')
    ax2.plot(pred_aligned[:, 0], pred_aligned[:, 2], 'r-', linewidth=2, label='Predicted')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('XZ Trajectory')
    ax2.legend()
    ax2.grid(True)
    
    # Error plot
    n = min(len(gt_aligned), len(pred_aligned))
    errors = np.linalg.norm(gt_aligned[:n] - pred_aligned[:n], axis=1)
    
    ax3 = axes[1, 0]
    ax3.plot(errors, 'purple', linewidth=2)
    ax3.axhline(errors.mean(), color='orange', linestyle='--', linewidth=2, 
                label=f'Mean {errors.mean():.3f} m')
    ax3.set_xlabel('Pose Index')
    ax3.set_ylabel('Error (m)')
    ax3.set_title('Point-wise Error')
    ax3.legend()
    ax3.grid(True)
    
    # Error histogram
    ax4 = axes[1, 1]
    ax4.hist(errors, bins=20, color='lightgray', edgecolor='black', alpha=0.7)
    ax4.axvline(errors.mean(), color='orange', linestyle='--', linewidth=2, 
                label=f'Mean {errors.mean():.3f} m')
    ax4.axvline(np.median(errors), color='red', linestyle='--', linewidth=2, 
                label=f'Median {np.median(errors):.3f} m')
    ax4.set_xlabel('Error (m)')
    ax4.set_ylabel('Count')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f'camera_{camera_name}_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved plot: {filename}")
    
    plt.show()  # Show interactive plots

def plot_all_trajectories(all_results, save_plots=False):
    """
    Plot all camera trajectories together.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('All Cameras Trajectory Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # XY plot
    ax1 = axes[0, 0]
    for i, (camera_name, data) in enumerate(all_results.items()):
        gt_aligned = data['gt_aligned']
        pred_aligned = data['pred_aligned']
        color = colors[i % len(colors)]
        
        ax1.plot(gt_aligned[:, 0], gt_aligned[:, 1], '-', color=color, 
                linewidth=2, label=f'GT Cam {camera_name}', alpha=0.7)
        ax1.plot(pred_aligned[:, 0], pred_aligned[:, 1], '--', color=color, 
                linewidth=2, label=f'Pred Cam {camera_name}', alpha=0.7)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectories - All Cameras')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # XZ plot
    ax2 = axes[0, 1]
    for i, (camera_name, data) in enumerate(all_results.items()):
        gt_aligned = data['gt_aligned']
        pred_aligned = data['pred_aligned']
        color = colors[i % len(colors)]
        
        ax2.plot(gt_aligned[:, 0], gt_aligned[:, 2], '-', color=color, 
                linewidth=2, label=f'GT Cam {camera_name}', alpha=0.7)
        ax2.plot(pred_aligned[:, 0], pred_aligned[:, 2], '--', color=color, 
                linewidth=2, label=f'Pred Cam {camera_name}', alpha=0.7)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('XZ Trajectories - All Cameras')
    ax2.legend()
    ax2.grid(True)
    
    # ATE comparison bar chart
    ax3 = axes[1, 0]
    camera_names = list(all_results.keys())
    ate_means = [all_results[cam]['ate_metrics']['mean'] for cam in camera_names]
    ate_stds = [all_results[cam]['ate_metrics']['std'] for cam in camera_names]
    
    bars = ax3.bar(camera_names, ate_means, yerr=ate_stds, capsize=5, 
                   color=colors[:len(camera_names)], alpha=0.7)
    ax3.set_ylabel('ATE Mean (m)')
    ax3.set_title('ATE Comparison Across Cameras')
    ax3.grid(True, axis='y')
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, ate_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    # Combined error distribution
    ax4 = axes[1, 1]
    all_errors = []
    for camera_name, data in all_results.items():
        gt_aligned = data['gt_aligned']
        pred_aligned = data['pred_aligned']
        n = min(len(gt_aligned), len(pred_aligned))
        errors = np.linalg.norm(gt_aligned[:n] - pred_aligned[:n], axis=1)
        all_errors.extend(errors)
    
    ax4.hist(all_errors, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, 
                label=f'Overall Mean {np.mean(all_errors):.3f} m')
    ax4.set_xlabel('Error (m)')
    ax4.set_ylabel('Count')
    ax4.set_title('Combined Error Distribution')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('all_cameras_comparison.png', dpi=300, bbox_inches='tight')
        print("  Saved plot: all_cameras_comparison.png")
    
    plt.show()  # Show interactive plots

def main():
    """Main function to compare multi-camera trajectories."""
    
    print("=== MULTI-CAMERA TRAJECTORY COMPARISON TOOL ===")
    
    # Define camera names
    camera_names = ['2']
    
    # Storage for all results
    all_results = {}
    all_ate_metrics = {}
    apple = True
    agri_orchard = "2d-apple_" if apple else "3d-pear"
    agri_stage = "dormancy"
    agri_scene = f"{agri_orchard}{agri_stage}"
    
    # Process each camera
    for camera_name in camera_names:
        gt_file = f'agri-data/3d-pear_{agri_stage}_train_c10_l20/groundtruth_cam_2.csv'
        pred_file = f'pings_experiments/3d-pear_{agri_stage}_train_c10_l20/trajectory.txt'
        
        # Check if files exist
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth file {gt_file} not found, skipping camera {camera_name}")
            continue
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file {pred_file} not found, skipping camera {camera_name}")
            continue
        
        try:
            # Load trajectories
            print(f"\nLoading camera {camera_name} trajectories...")
            gt_traj = parse_csv_trajectory(gt_file)
            pred_traj = parse_tum_trajectory(pred_file)
            
            print(f"✓ Loaded GT trajectory with {len(gt_traj['timestamps'])} poses")
            print(f"✓ Loaded predicted trajectory with {len(pred_traj['positions'])} poses")
            
            # Compare trajectories
            ate_metrics, ate_metrics_xy, gt_aligned, pred_aligned, gt_xy, pred_xy_umeyama = compare_single_trajectory(
                gt_traj, pred_traj, camera_name)
            
            # Store results
            all_results[camera_name] = {
                'ate_metrics': ate_metrics,
                'ate_metrics_xy': ate_metrics_xy,
                'gt_aligned': gt_aligned,
                'pred_aligned': pred_aligned,
                'gt_xy': gt_xy,
                'pred_xy_umeyama': pred_xy_umeyama
            }
            all_ate_metrics[camera_name] = ate_metrics
            
            # Plot individual camera results with both 3D and 2D comparisons
            plot_camera_trajectory_enhanced(gt_aligned, pred_aligned, gt_xy, pred_xy_umeyama, 
                                          ate_metrics, ate_metrics_xy, camera_name, save_plots=False)
            
        except Exception as e:
            print(f"✗ Error processing camera {camera_name}: {e}")
            continue
    
    if not all_ate_metrics:
        print("No cameras were successfully processed!")
        return
    
    # Plot combined results
    plot_all_trajectories(all_results, save_plots=True)
    
    # Calculate and display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n{'Camera':<8} {'3D Mean ATE':<12} {'3D RMSE':<12} {'2D XY Mean':<12} {'2D XY RMSE':<12} {'Poses':<8}")
    print("-" * 80)
    
    all_means_3d = []
    all_rmses_3d = []
    all_means_2d = []
    all_rmses_2d = []
    total_poses = 0
    
    for camera_name in sorted(all_ate_metrics.keys()):
        metrics_3d = all_ate_metrics[camera_name]
        metrics_2d = all_results[camera_name]['ate_metrics_xy']
        print(f"Cam {camera_name:<4} {metrics_3d['mean']:<12.4f} {metrics_3d['rmse']:<12.4f} "
              f"{metrics_2d['mean']:<12.4f} {metrics_2d['rmse']:<12.4f} {metrics_3d['num_poses']:<8}")
        all_means_3d.append(metrics_3d['mean'])
        all_rmses_3d.append(metrics_3d['rmse'])
        all_means_2d.append(metrics_2d['mean'])
        all_rmses_2d.append(metrics_2d['rmse'])
        total_poses += metrics_3d['num_poses']
    
    print("-" * 80)
    print(f"Average: {np.mean(all_means_3d):<12.4f} {np.mean(all_rmses_3d):<12.4f} "
          f"{np.mean(all_means_2d):<12.4f} {np.mean(all_rmses_2d):<12.4f} {total_poses:<8}")
    
    print(f"\n=== OVERALL RESULTS ===")
    print(f"Number of cameras processed: {len(all_ate_metrics)}")
    print(f"3D Average ATE across all cameras: {np.mean(all_means_3d):.4f} ± {np.std(all_means_3d):.4f} m")
    print(f"3D Average RMSE across all cameras: {np.mean(all_rmses_3d):.4f} ± {np.std(all_rmses_3d):.4f} m")
    print(f"2D XY Average ATE across all cameras: {np.mean(all_means_2d):.4f} ± {np.std(all_means_2d):.4f} m")
    print(f"2D XY Average RMSE across all cameras: {np.mean(all_rmses_2d):.4f} ± {np.std(all_rmses_2d):.4f} m")
    print(f"Best performing camera (3D): Cam {min(all_ate_metrics.keys(), key=lambda x: all_ate_metrics[x]['mean'])} "
          f"(ATE: {min(all_means_3d):.4f} m)")
    print(f"Best performing camera (2D XY): Cam {min(all_results.keys(), key=lambda x: all_results[x]['ate_metrics_xy']['mean'])} "
          f"(ATE: {min(all_means_2d):.4f} m)")
    print(f"Improvement with 2D XY Umeyama: {((np.mean(all_means_3d) - np.mean(all_means_2d)) / np.mean(all_means_3d) * 100):+.2f}%")

if __name__ == "__main__":
    main()
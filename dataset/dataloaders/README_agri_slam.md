# AgriSLAM Dataset for PINGS

This directory contains a custom dataset loader for agricultural robotics data compatible with the PINGS system.

## Dataset Structure

The AgriSLAM dataset uses data from `/packages/pings/agri-slam-data/train/` with the following structure:

```
agri-slam-data/train/
├── ouster/points/          # Ouster LiDAR point clouds (.ply files)
├── zed_multi/cam_2/rgb/    # ZED Camera 2 RGB images (.jpg files)
├── groundtruth_lidar.csv   # LiDAR ground truth poses
├── groundtruth_cam_2.csv   # Camera 2 ground truth poses
└── fixposition/            # Additional positioning data (not used)
```

## Data Details

- **281 LiDAR scans** in PLY format from Ouster sensor
- **153 RGB images** from ZED Camera 2 in 1920x1200 resolution  
- **Ground truth poses** in CSV format with timestamp, translation (tx,ty,tz), and quaternion (qx,qy,qz,qw)
- **Coordinate system**: The poses are in a global coordinate frame (appears to be UTM)

## Key Features

- **NaN filtering**: Invalid LiDAR returns (NaN values) are automatically filtered out
- **Timestamp synchronization**: Automatic matching between LiDAR and camera timestamps
- **Ground truth poses**: Both LiDAR and camera poses available for evaluation
- **Calibration ready**: Camera intrinsics and extrinsics can be customized

## Configuration

The dataset uses `config/run_agri_slam_gs.yaml` which is optimized for:
- Agricultural outdoor environments
- Longer range sensing (up to 80m)
- Gaussian surfel representation for better outdoor performance
- Disabled monodepth initially to avoid dependency issues

## Usage Examples

### Basic SLAM run:
```bash
python3 pings.py ./config/run_agri_slam_gs.yaml agri_slam train --visualize --save-map --log-on
```

### Quick test with limited frames:
```bash
python3 pings.py ./config/run_agri_slam_gs.yaml agri_slam train --range 0 50 1 --visualize --log-on
```

### Save all outputs:
```bash
python3 pings.py ./config/run_agri_slam_gs.yaml agri_slam train --save-map --save-mesh --save-merged-pc --log-on
```

## Implementation Notes

- **Hardcoded paths**: The dataset loader uses hardcoded paths to `/packages/pings/agri-slam-data/train/`
- **Point cloud format**: PLY files are loaded using Open3D and converted to (N,4) arrays with intensity
- **Image format**: JPG images are loaded and converted to RGB format
- **Pose format**: CSV poses are converted to 4x4 transformation matrices
- **Camera calibration**: Uses approximate ZED camera intrinsics (should be calibrated for your setup)

## Customization

To adapt this dataset for your own data:

1. **Update paths**: Modify the hardcoded paths in `AgriSLAMDataset.__init__()`
2. **Camera calibration**: Update `self.K_cam` and `self.T_cam_lidar` with your calibration
3. **File formats**: Modify `read_point_cloud()` and `read_img()` for different formats
4. **Timestamps**: Adjust `_create_timestamp_mappings()` for your timestamp format

## Files

- `agri_slam.py` - Main dataset loader class
- `run_agri_slam_gs.yaml` - Configuration file optimized for this dataset
- `test_agri_slam.py` - Test script to verify dataset loading
- `run_agri_slam_example.py` - Usage examples and instructions

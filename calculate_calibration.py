import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pose):
    """Convert [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix"""
    x, y, z, qx, qy, qz, qw = pose
    
    # Create rotation matrix from quaternion
    rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = [x, y, z]
    
    return T

# Camera 2 extrinsic (cam_2 to world frame)
cam2_extrinsic = [0.14374693, 0.06611648, -0.58771134, -0.49272718, 0.51430475, -0.50217014, 0.49044439]
T_world_cam2 = pose_to_matrix(cam2_extrinsic)

# LiDAR extrinsic (LiDAR to world frame)  
lidar_extrinsic = [-0.02773899, -0.02091633, -0.18058062, 0.36610677, 0.93049435, 0.01022933, 0.00643797]
T_world_lidar = pose_to_matrix(lidar_extrinsic)

# Calculate T_cam2_lidar = T_cam2_world * T_world_lidar
T_cam2_world = np.linalg.inv(T_world_cam2)
T_cam2_lidar = T_cam2_world @ T_world_lidar

# Camera 2 intrinsics [fx, fy, cx, cy]
fx, fy, cx, cy = 737.81004, 737.95291, 975.08049, 569.79751

# Create K matrix
K_cam2 = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy], 
    [0.0, 0.0, 1.0]
])

print("Camera 2 Intrinsic Matrix (K_cam2):")
print(K_cam2)
print()

print("Camera 2 to LiDAR Transformation Matrix (T_cam2_lidar):")
print(T_cam2_lidar)
print()

print("For copying into your code:")
print("K_cam = np.array([")
print(f"    [{fx}, 0.0, {cx}],")
print(f"    [0.0, {fy}, {cy}],") 
print(f"    [0.0, 0.0, 1.0]")
print("])")
print()

print("T_cam_lidar = np.array([")
for i in range(4):
    row_str = "    [" + ", ".join([f"{T_cam2_lidar[i,j]:.8f}" for j in range(4)]) + "]"
    if i < 3:
        row_str += ","
    print(row_str)
print("])")

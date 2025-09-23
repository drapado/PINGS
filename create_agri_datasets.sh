#!/bin/bash

# Script to create train and val datasets for 2d-apples and 3d-pears
# with three states (dormancy, flowering, harvesting) from /data folder
# Copies zed_multi/cam_2, ouster folder, groundtruth_cam_2.csv, and groundtruth_lidar.csv files

set -e  # Exit on any error

# Source data directory
SOURCE_DIR="/media/david/t7_4tb/agri-slam/agri-gs-slam-dataset/"

# Target directory
TARGET_DIR="agri-data"

# Crop types
CROP_TYPES=("2d-apple" "3d-pear")

# States
STATES=("dormancy" "flowering" "harvesting")

# Dataset splits
SPLITS=("train" "val")

echo "Creating AGRI datasets..."
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo ""

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Function to copy data for a specific configuration
copy_data() {
    local crop_type=$1
    local state=$2
    local split=$3
    
    local source_path="$SOURCE_DIR/$crop_type/$state/$split"
    local target_path="$TARGET_DIR/${crop_type}_${state}_${split}"
    
    echo "Processing: $crop_type -> $state -> $split"
    
    # Check if source directory exists
    if [ ! -d "$source_path" ]; then
        echo "  WARNING: Source directory does not exist: $source_path"
        return
    fi
    
    # Check if required files/directories exist
    local cam_2_path="$source_path/zed_multi/cam_2"
    local ouster_path="$source_path/ouster"
    local groundtruth_cam_path="$source_path/groundtruth_cam_2.csv"
    local groundtruth_lidar_path="$source_path/groundtruth_lidar.csv"
    
    if [ ! -d "$cam_2_path" ]; then
        echo "  WARNING: cam_2 directory not found: $cam_2_path"
        return
    fi
    
    if [ ! -d "$ouster_path" ]; then
        echo "  WARNING: ouster directory not found: $ouster_path"
        return
    fi
    
    if [ ! -f "$groundtruth_cam_path" ]; then
        echo "  WARNING: groundtruth_cam_2.csv not found: $groundtruth_cam_path"
        return
    fi
    
    if [ ! -f "$groundtruth_lidar_path" ]; then
        echo "  WARNING: groundtruth_lidar.csv not found: $groundtruth_lidar_path"
        return
    fi
    
    # Create target directory
    mkdir -p "$target_path"
    mkdir -p "$target_path/zed_multi"
    
    # Copy cam_2 directory with progress (excluding hidden files)
    echo "  Copying cam_2 data..."
    rsync -ah --progress --exclude='.*' "$cam_2_path" "$target_path/zed_multi/" 2>/dev/null || {
        mkdir -p "$target_path/zed_multi/cam_2"
        find "$cam_2_path" -type f ! -name '.*' -exec cp {} "$target_path/zed_multi/cam_2/" \;
    }
    
    # Copy ouster directory with progress (excluding hidden files)
    echo "  Copying ouster data..."
    rsync -ah --progress --exclude='.*' "$ouster_path" "$target_path/" 2>/dev/null || {
        mkdir -p "$target_path/ouster"
        find "$ouster_path" -type f ! -name '.*' -exec cp {} "$target_path/ouster/" \;
    }
    
    # Copy groundtruth files
    echo "  Copying groundtruth_cam_2.csv..."
    cp "$groundtruth_cam_path" "$target_path/"
    
    echo "  Copying groundtruth_lidar.csv..."
    cp "$groundtruth_lidar_path" "$target_path/"
    
    echo "  ✓ Completed: $target_path"
    echo ""
}

# Main processing loop
for crop_type in "${CROP_TYPES[@]}"; do
    echo "=== Processing $crop_type ==="
    
    for state in "${STATES[@]}"; do
        echo "--- State: $state ---"
        
        for split in "${SPLITS[@]}"; do
            copy_data "$crop_type" "$state" "$split"
        done
    done
done

echo "=== Summary ==="
echo "Dataset creation completed!"
echo ""
echo "Created datasets in $TARGET_DIR:"
ls -la "$TARGET_DIR" 2>/dev/null || echo "No datasets created (check warnings above)"

echo ""
echo "Script finished successfully!"

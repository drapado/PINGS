#!/bin/bash

# Script to downsample a dataset by a specified factor
# Creates a new dataset with the downsample factor in the directory name

# Function to display usage
usage() {
    echo "Usage: $0 <dataset_dir> <downsample_factor>"
    echo ""
    echo "Arguments:"
    echo "  dataset_dir      Path to dataset directory (e.g., agri-data/2d-apple_harvesting_train)"
    echo "  downsample_factor Downsampling factor (must be > 1)"
    echo ""
    echo "Example:"
    echo "  $0 agri-data/2d-apple_harvesting_train 5"
    echo ""
    echo "This will create a downsampled dataset with name ending in '_5'"
    exit 1
}

# Function to copy and downsample images from a directory
copy_images_with_downsampling() {
    local src_dir="$1"
    local dst_dir="$2"
    local downsample_factor="$3"
    
    if [ ! -d "$src_dir" ]; then
        echo "Warning: Source directory $src_dir does not exist"
        return 0
    fi
    
    mkdir -p "$dst_dir"
    
    local count=0
    local copied=0
    
    # Sort files by name to ensure consistent ordering
    # Handle both .png and .jpg files
    for file in $(ls "$src_dir"/*.png "$src_dir"/*.jpg 2>/dev/null | sort); do
        if [ $((count % downsample_factor)) -eq 0 ]; then
            local filename=$(basename "$file")
            cp "$file" "$dst_dir/"
            ((copied++))
        fi
        ((count++))
    done
    
    echo "$copied"
}

# Function to copy and downsample any files from a directory
copy_files_with_downsampling() {
    local src_dir="$1"
    local dst_dir="$2"
    local downsample_factor="$3"
    
    if [ ! -d "$src_dir" ]; then
        echo "Warning: Source directory $src_dir does not exist"
        return 0
    fi
    
    mkdir -p "$dst_dir"
    
    local count=0
    local copied=0
    
    # Sort files by name to ensure consistent ordering
    # Handle any file type (excluding hidden files and directories)
    for file in $(find "$src_dir" -maxdepth 1 -type f ! -name ".*" | sort); do
        if [ $((count % downsample_factor)) -eq 0 ]; then
            local filename=$(basename "$file")
            cp "$file" "$dst_dir/"
            ((copied++))
        fi
        ((count++))
    done
    
    echo "$copied"
}

# Function to downsample CSV file
downsample_csv() {
    local input_csv="$1"
    local output_csv="$2"
    local downsample_factor="$3"
    
    if [ ! -f "$input_csv" ]; then
        echo "Warning: CSV file $input_csv does not exist"
        return
    fi
    
    # Copy header
    head -n 1 "$input_csv" > "$output_csv"
    
    # Downsample data rows
    local count=0
    tail -n +2 "$input_csv" | while read line; do
        if [ $((count % downsample_factor)) -eq 0 ]; then
            echo "$line" >> "$output_csv"
        fi
        ((count++))
    done
}

# Check arguments
if [ $# -ne 2 ]; then
    usage
fi

DATASET_DIR="$1"
DOWNSAMPLE_FACTOR="$2"

# Validate inputs
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' does not exist"
    exit 1
fi

if ! [[ "$DOWNSAMPLE_FACTOR" =~ ^[0-9]+$ ]] || [ "$DOWNSAMPLE_FACTOR" -lt 2 ]; then
    echo "Error: Downsample factor must be an integer >= 2"
    exit 1
fi

# Extract base name for downsampled dataset
DATASET_BASE=$(basename "$DATASET_DIR")
DOWNSAMPLED_NAME="${DATASET_BASE}_${DOWNSAMPLE_FACTOR}"
DOWNSAMPLED_DIR="$(dirname "$DATASET_DIR")/$DOWNSAMPLED_NAME"

echo "Creating downsampled dataset: $DOWNSAMPLED_DIR"
echo "Downsample factor: $DOWNSAMPLE_FACTOR"

# Remove existing downsampled directory if it exists
if [ -d "$DOWNSAMPLED_DIR" ]; then
    echo "Removing existing directory: $DOWNSAMPLED_DIR"
    rm -rf "$DOWNSAMPLED_DIR"
fi

# Create directory structure
mkdir -p "$DOWNSAMPLED_DIR/zed_multi/cam_2"
mkdir -p "$DOWNSAMPLED_DIR/ouster/imu"
mkdir -p "$DOWNSAMPLED_DIR/ouster/points"

# Define subdirectories to process
SUBDIRS=("rgb" "depth" "depth_anything" "point_cloud")

echo ""
echo "Processing dataset..."

# Process data
total_images=0
for subdir in "${SUBDIRS[@]}"; do
    src_subdir="$DATASET_DIR/zed_multi/cam_2/$subdir"
    dst_subdir="$DOWNSAMPLED_DIR/zed_multi/cam_2/$subdir"
    
    if [ -d "$src_subdir" ]; then
        echo "  Processing $subdir..."
        if [ "$subdir" = "point_cloud" ]; then
            copied=$(copy_files_with_downsampling "$src_subdir" "$dst_subdir" "$DOWNSAMPLE_FACTOR")
        else
            copied=$(copy_images_with_downsampling "$src_subdir" "$dst_subdir" "$DOWNSAMPLE_FACTOR")
        fi
        if [ "$subdir" = "rgb" ]; then
            total_images=$copied
        fi
        echo "    Copied $copied files from $subdir"
    fi
done

# Process ouster data
ouster_src_dir="$DATASET_DIR/ouster"
ouster_dst_dir="$DOWNSAMPLED_DIR/ouster"

if [ -d "$ouster_src_dir" ]; then
    echo "  Processing ouster data..."
    
    # Copy map.ply if it exists
    if [ -f "$ouster_src_dir/map.ply" ]; then
        cp "$ouster_src_dir/map.ply" "$ouster_dst_dir/"
        echo "    Copied map.ply"
    fi
    
    # Process points folder with downsampling
    if [ -d "$ouster_src_dir/points" ]; then
        copied=$(copy_files_with_downsampling "$ouster_src_dir/points" "$ouster_dst_dir/points" "$DOWNSAMPLE_FACTOR")
        echo "    Copied $copied point files from ouster/points"
    fi
fi

echo ""
echo "Processing CSV files..."

# Process CSV files with downsampling
if [ -f "$DATASET_DIR/groundtruth_cam_2.csv" ]; then
    echo "  Processing camera CSV..."
    downsample_csv "$DATASET_DIR/groundtruth_cam_2.csv" "$DOWNSAMPLED_DIR/groundtruth_cam_2.csv" "$DOWNSAMPLE_FACTOR"
    echo "    Created: groundtruth_cam_2.csv"
fi

if [ -f "$DATASET_DIR/groundtruth_lidar.csv" ]; then
    echo "  Processing lidar CSV..."
    downsample_csv "$DATASET_DIR/groundtruth_lidar.csv" "$DOWNSAMPLED_DIR/groundtruth_lidar.csv" "$DOWNSAMPLE_FACTOR"
    echo "    Created: groundtruth_lidar.csv"
fi

echo ""
echo "=================================="
echo "DATASET DOWNSAMPLING COMPLETE"
echo "=================================="
echo "Original dataset: $DATASET_DIR"
echo "Downsampled dataset: $DOWNSAMPLED_DIR"
echo "Total images: $total_images"
echo "Downsample factor: $DOWNSAMPLE_FACTOR"
echo ""
echo "Files created:"
if [ -f "$DOWNSAMPLED_DIR/groundtruth_cam_2.csv" ]; then
    echo "  - groundtruth_cam_2.csv (downsampled camera poses)"
fi
if [ -f "$DOWNSAMPLED_DIR/groundtruth_lidar.csv" ]; then
    echo "  - groundtruth_lidar.csv (downsampled lidar poses)"
fi
echo "=================================="
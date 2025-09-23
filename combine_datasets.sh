#!/bin/bash

# Script to combine train and validation datasets into a single train_val dataset
# with optional downsampling

# Function to display usage
usage() {
    echo "Usage: $0 <train_dir> <val_dir> [downsample_factor]"
    echo ""
    echo "Arguments:"
    echo "  train_dir        Path to training dataset directory (e.g., datasets/agri/2d-apple_harvesting_train)"
    echo "  val_dir          Path to validation dataset directory (e.g., datasets/agri/2d-apple_harvesting_val)"
    echo "  downsample_factor Optional downsampling factor (default: 1, no downsampling)"
    echo ""
    echo "Example:"
    echo "  $0 datasets/agri/2d-apple_harvesting_train datasets/agri/2d-apple_harvesting_val 2"
    echo ""
    echo "This will create a combined dataset with name ending in '_train_val'"
    exit 1
}

# Function to copy and downsample images from a directory
copy_images_with_downsampling() {
    local src_dir="$1"
    local dst_dir="$2"
    local downsample_factor="$3"
    local start_index="$4"
    
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
    local start_index="$4"
    
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
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    usage
fi

TRAIN_DIR="$1"
VAL_DIR="$2"
DOWNSAMPLE_FACTOR="${3:-1}"

# Validate inputs
if [ ! -d "$TRAIN_DIR" ]; then
    echo "Error: Training directory '$TRAIN_DIR' does not exist"
    exit 1
fi

if [ ! -d "$VAL_DIR" ]; then
    echo "Error: Validation directory '$VAL_DIR' does not exist"
    exit 1
fi

if ! [[ "$DOWNSAMPLE_FACTOR" =~ ^[0-9]+$ ]] || [ "$DOWNSAMPLE_FACTOR" -lt 1 ]; then
    echo "Error: Downsample factor must be a positive integer"
    exit 1
fi

# Extract base name for combined dataset
TRAIN_BASE=$(basename "$TRAIN_DIR")
COMBINED_NAME="${TRAIN_BASE%_train}_train_val"
COMBINED_DIR="$(dirname "$TRAIN_DIR")/$COMBINED_NAME"

echo "Creating combined dataset: $COMBINED_DIR"
echo "Downsample factor: $DOWNSAMPLE_FACTOR"

# Remove existing combined directory if it exists
if [ -d "$COMBINED_DIR" ]; then
    echo "Removing existing directory: $COMBINED_DIR"
    rm -rf "$COMBINED_DIR"
fi

# Create directory structure
mkdir -p "$COMBINED_DIR/zed_multi/cam_2"
mkdir -p "$COMBINED_DIR/ouster/imu"
mkdir -p "$COMBINED_DIR/ouster/points"

# Define subdirectories to process
SUBDIRS=("rgb" "depth" "depth_anything" "point_cloud")

echo ""
echo "Processing train dataset..."

# Process train data
total_train_images=0
for subdir in "${SUBDIRS[@]}"; do
    train_subdir="$TRAIN_DIR/zed_multi/cam_2/$subdir"
    combined_subdir="$COMBINED_DIR/zed_multi/cam_2/$subdir"
    
    if [ -d "$train_subdir" ]; then
        echo "  Processing $subdir..."
        if [ "$subdir" = "point_cloud" ]; then
            copied=$(copy_files_with_downsampling "$train_subdir" "$combined_subdir" "$DOWNSAMPLE_FACTOR" 0)
        else
            copied=$(copy_images_with_downsampling "$train_subdir" "$combined_subdir" "$DOWNSAMPLE_FACTOR" 0)
        fi
        if [ "$subdir" = "rgb" ]; then
            total_train_images=$copied
        fi
        echo "    Copied $copied images from train $subdir"
    fi
done

# Process ouster data from train
ouster_train_dir="$TRAIN_DIR/ouster"
ouster_combined_dir="$COMBINED_DIR/ouster"

if [ -d "$ouster_train_dir" ]; then
    echo "  Processing ouster data..."
    
    # Copy map.ply if it exists
    if [ -f "$ouster_train_dir/map.ply" ]; then
        cp "$ouster_train_dir/map.ply" "$ouster_combined_dir/"
        echo "    Copied map.ply from train ouster"
    fi
    
    # Process points folder with downsampling
    if [ -d "$ouster_train_dir/points" ]; then
        copied=$(copy_files_with_downsampling "$ouster_train_dir/points" "$ouster_combined_dir/points" "$DOWNSAMPLE_FACTOR" 0)
        echo "    Copied $copied point files from train ouster/points"
    fi
fi

echo ""
echo "Processing validation dataset..."

# Process validation data
total_val_images=0
for subdir in "${SUBDIRS[@]}"; do
    val_subdir="$VAL_DIR/zed_multi/cam_2/$subdir"
    combined_subdir="$COMBINED_DIR/zed_multi/cam_2/$subdir"
    
    if [ -d "$val_subdir" ]; then
        echo "  Processing $subdir..."
        if [ "$subdir" = "point_cloud" ]; then
            copied=$(copy_files_with_downsampling "$val_subdir" "$combined_subdir" "$DOWNSAMPLE_FACTOR" "$total_train_images")
        else
            copied=$(copy_images_with_downsampling "$val_subdir" "$combined_subdir" "$DOWNSAMPLE_FACTOR" "$total_train_images")
        fi
        if [ "$subdir" = "rgb" ]; then
            total_val_images=$copied
        fi
        echo "    Copied $copied images from val $subdir"
    fi
done

# Process ouster data from validation
ouster_val_dir="$VAL_DIR/ouster"

if [ -d "$ouster_val_dir" ]; then
    echo "  Processing ouster data..."
    
    # Copy map.ply if it exists (validation map.ply will overwrite train map.ply)
    if [ -f "$ouster_val_dir/map.ply" ]; then
        cp "$ouster_val_dir/map.ply" "$ouster_combined_dir/map_val.ply"
        echo "    Copied map.ply from val ouster as map_val.ply"
    fi
    
    # Process points folder with downsampling
    if [ -d "$ouster_val_dir/points" ]; then
        copied=$(copy_files_with_downsampling "$ouster_val_dir/points" "$ouster_combined_dir/points" "$DOWNSAMPLE_FACTOR" "$total_train_images")
        echo "    Copied $copied point files from val ouster/points"
    fi
fi

echo ""
echo "Processing CSV files..."

# Process CSV files - copy both train and val groundtruth files
if [ -f "$TRAIN_DIR/groundtruth_cam_2.csv" ]; then
    echo "  Processing train CSV..."
    downsample_csv "$TRAIN_DIR/groundtruth_cam_2.csv" "$COMBINED_DIR/groundtruth_cam_2_train.csv" "1"
    echo "    Created: groundtruth_cam_2_train.csv"
fi

if [ -f "$VAL_DIR/groundtruth_cam_2.csv" ]; then
    echo "  Processing validation CSV..."
    downsample_csv "$VAL_DIR/groundtruth_cam_2.csv" "$COMBINED_DIR/groundtruth_cam_2_val.csv" "1"
    echo "    Created: groundtruth_cam_2_val.csv"
fi

# Create combined CSV file
if [ -f "$COMBINED_DIR/groundtruth_cam_2_train.csv" ] && [ -f "$COMBINED_DIR/groundtruth_cam_2_val.csv" ]; then
    echo "  Creating combined CSV..."
    # Copy header from train CSV
    head -n 1 "$COMBINED_DIR/groundtruth_cam_2_train.csv" > "$COMBINED_DIR/groundtruth_cam_2.csv"
    # Add train data (without header)
    tail -n +2 "$COMBINED_DIR/groundtruth_cam_2_train.csv" >> "$COMBINED_DIR/groundtruth_cam_2.csv"
    # Add val data (without header)
    tail -n +2 "$COMBINED_DIR/groundtruth_cam_2_val.csv" >> "$COMBINED_DIR/groundtruth_cam_2.csv"
    echo "    Created: groundtruth_cam_2.csv (combined)"
fi

# Process lidar CSV files - copy both train and val groundtruth files
if [ -f "$TRAIN_DIR/groundtruth_lidar.csv" ]; then
    echo "  Processing train lidar CSV..."
    downsample_csv "$TRAIN_DIR/groundtruth_lidar.csv" "$COMBINED_DIR/groundtruth_lidar_train.csv" "1"
    echo "    Created: groundtruth_lidar_train.csv"
fi

if [ -f "$VAL_DIR/groundtruth_lidar.csv" ]; then
    echo "  Processing validation lidar CSV..."
    downsample_csv "$VAL_DIR/groundtruth_lidar.csv" "$COMBINED_DIR/groundtruth_lidar_val.csv" "1"
    echo "    Created: groundtruth_lidar_val.csv"
fi

# Create combined lidar CSV file
if [ -f "$COMBINED_DIR/groundtruth_lidar_train.csv" ] && [ -f "$COMBINED_DIR/groundtruth_lidar_val.csv" ]; then
    echo "  Creating combined lidar CSV..."
    # Copy header from train CSV
    head -n 1 "$COMBINED_DIR/groundtruth_lidar_train.csv" > "$COMBINED_DIR/groundtruth_lidar.csv"
    # Add train data (without header)
    tail -n +2 "$COMBINED_DIR/groundtruth_lidar_train.csv" >> "$COMBINED_DIR/groundtruth_lidar.csv"
    # Add val data (without header)
    tail -n +2 "$COMBINED_DIR/groundtruth_lidar_val.csv" >> "$COMBINED_DIR/groundtruth_lidar.csv"
    echo "    Created: groundtruth_lidar.csv (combined)"
fi

# Calculate validation start frame
val_start_frame=$((total_train_images + 1))

echo ""
echo "=================================="
echo "DATASET COMBINATION COMPLETE"
echo "=================================="
echo "Combined dataset location: $COMBINED_DIR"
echo "Train images: $total_train_images"
echo "Validation images: $total_val_images"
echo "Total images: $((total_train_images + total_val_images))"
echo "Downsample factor: $DOWNSAMPLE_FACTOR"
echo ""
echo "VALIDATION START FRAME: $val_start_frame"
echo ""
echo "Files created:"
echo "  - groundtruth_cam_2.csv (combined poses)"
echo "  - groundtruth_cam_2_train.csv (train poses only)"
echo "  - groundtruth_cam_2_val.csv (validation poses only)"
echo "  - groundtruth_lidar.csv (combined lidar poses)"
echo "  - groundtruth_lidar_train.csv (train lidar poses only)"
echo "  - groundtruth_lidar_val.csv (validation lidar poses only)"
echo "=================================="

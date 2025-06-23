#!/bin/bash
#SBATCH --job-name=copy_selected_folders
#SBATCH --output=copy_selected_folders_%j.out
#SBATCH --error=copy_selected_folders_%j.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=high
#SBATCH --account=billengrp

# Source and destination directories
SRC_DIR="/nfs/peloton/waltz/billenlab/mpi-io-DOES-NOT-WORK-ON-HIVE/lochy"
DEST_DIR="/quobyte/billengrp/lochy"

# List of folders to copy; use ("*") to copy everything
FOLDERS_TO_COPY=("ThDSubduction" "TwoDSubduction" "ThDSlab")  # Or ("*") to copy all

# Choose copy method: "scp" or "rsync"
COPY_METHOD="rsync"  # or "scp"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Choose which tool to use
copy_folder() {
    local src_path="$1"
    local dest_path="$2"

    if [ "$COPY_METHOD" == "scp" ]; then
        echo "  -> Using: scp -r $src_path $dest_path"
        scp -r "$src_path" "$dest_path"
    elif [ "$COPY_METHOD" == "rsync" ]; then
        echo "  -> Using: rsync -avu --progress $src_path $dest_path"
        rsync -avu --progress "$src_path" "$dest_path"
    else
        echo "ERROR: Unsupported COPY_METHOD: $COPY_METHOD"
        exit 1
    fi
}

# Perform the copy
if [ "${#FOLDERS_TO_COPY[@]}" -eq 1 ] && [ "${FOLDERS_TO_COPY[0]}" = "*" ]; then
    echo "Copying all contents from $SRC_DIR to $DEST_DIR using $COPY_METHOD"
    for item in "$SRC_DIR"/*; do
        [ -e "$item" ] || continue
        copy_folder "$item" "$DEST_DIR/"
    done
else
    echo "Copying selected folders using $COPY_METHOD:"
    for FOLDER in "${FOLDERS_TO_COPY[@]}"; do
        echo "  - $FOLDER"
        if [ -d "$SRC_DIR/$FOLDER" ]; then
            copy_folder "$SRC_DIR/$FOLDER" "$DEST_DIR/"
        else
            echo "Warning: $SRC_DIR/$FOLDER does not exist"
        fi
    done
fi

echo "Copy job finished."
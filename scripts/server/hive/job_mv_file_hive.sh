#!/bin/bash
#SBATCH --job-name=mv_selected_folders
#SBATCH --output=mv_selected_folders_%j.out
#SBATCH --error=mv_selected_folders_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=high
#SBATCH --account=billengrp

#!/bin/bash

# set -euo pipefail

# Source and target directories
OLD_DIR="/nfs/peloton/waltz/billenlab/mpi-io-DOES-NOT-WORK-ON-HIVE/lochy/ThDSubduction"
NEW_DIR="/nfs/peloton/waltz/billenlab/group/lochy/ThDSubduction"

# Initialize counters
created_count=0
skipped_count=0

# Create base directory if it doesn't exist
mkdir -p "$NEW_DIR"

# Traverse all files and recreate hard links if not already present
find "$OLD_DIR" -type f | while read -r file; do
    rel_path="${file#$OLD_DIR/}"
    target_path="$NEW_DIR/$rel_path"
    target_dir="$(dirname "$target_path")"

    mkdir -p "$target_dir"

    if [[ -e "$target_path" ]]; then
        echo "Skipping existing file: $target_path"
        ((skipped_count++))
    else
        ln "$file" "$target_path"
        ((created_count++))
    fi
done

# Final summary
echo "Finished creating hard links."
echo "Links created: $created_count"
echo "Files skipped: $skipped_count"

echo ""
echo "===== Directory Size Summary ====="
echo "Source directory ($OLD_DIR):"
du -h -d2 "$OLD_DIR"

echo ""
echo "Target directory ($NEW_DIR):"
du -h -d2 "$NEW_DIR"

echo ""
echo "===== File Count Comparison ====="
count_old=$(find "$OLD_DIR" -type f | wc -l)
count_new=$(find "$NEW_DIR" -type f | wc -l)

echo "Files in source: $count_old"
echo "Files in target: $count_new"

if [[ "$count_old" -eq "$count_new" ]]; then
    echo "File counts match!"
else
    echo "File counts do NOT match!"
fi


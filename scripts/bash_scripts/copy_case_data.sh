#!/bin/bash

# Define paths
CASE_DIR="/mnt/lochz/ASPECT_DATA/TwoDSubduction/EBA_CDPT_morb_dE/eba_cdpt_coh500_SA80.0_cd7.5_log"
DEST_DIR="/mnt/lochz/ASPECT_DATA/TwoDSubduction/packed_data/Psa80oa40Rwedge"

# List of heat_flux steps
STEPS=(259 423 706)

# List of vtu snapshots
VTU_SNAPSHOTS=(29 70 113)

# Create destination directory and log file
mkdir -p "$DEST_DIR"
LOG_FILE="$DEST_DIR/file_list"
: > "$LOG_FILE"  # empty the log file

# Copy top-level case files
for FILE in case.prm case.wb case.json; do
    if [ -f "$CASE_DIR/$FILE" ]; then
        cp "$CASE_DIR/$FILE" "$DEST_DIR/"
        echo "$DEST_DIR/$FILE" >> "$LOG_FILE"
    fi
done

# Copy output/statistics folder
if [ -d "$CASE_DIR/output/statistics" ]; then
    mkdir -p "$DEST_DIR/output/statistics"
    for FILE in "$CASE_DIR/output/statistics/"*; do
        cp "$FILE" "$DEST_DIR/output/statistics/"
        echo "$DEST_DIR/output/statistics/$(basename "$FILE")" >> "$LOG_FILE"
    done
fi

# Copy output/solution.pvd
if [ -f "$CASE_DIR/output/solution.pvd" ]; then
    mkdir -p "$DEST_DIR/output"
    cp "$CASE_DIR/output/solution.pvd" "$DEST_DIR/output/"
    echo "$DEST_DIR/output/solution.pvd" >> "$LOG_FILE"
fi

# Copy selected heat_flux step files
for STEP in "${STEPS[@]}"; do
    FILENAME=$(printf "heat_flux.%05d" "$STEP")
    SRC="$CASE_DIR/output/$FILENAME"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DEST_DIR/output/"
        echo "$DEST_DIR/output/$FILENAME" >> "$LOG_FILE"
    else
        echo "Warning: $FILENAME not found in $CASE_DIR"
    fi
done

# Copy selected .pvtu and .vtu files
for SNAPSHOT in "${VTU_SNAPSHOTS[@]}"; do
    VTU_BASE=$(printf "solution-%05d" "$SNAPSHOT")
    SRC_DIR="$CASE_DIR/output/solution"
    DEST_SUBDIR="$DEST_DIR/output/solution"
    mkdir -p "$DEST_SUBDIR"

    # Copy .pvtu file
    PVTU="$VTU_BASE.pvtu"
    if [ -f "$SRC_DIR/$PVTU" ]; then
        cp "$SRC_DIR/$PVTU" "$DEST_SUBDIR/"
        echo "$DEST_SUBDIR/$PVTU" >> "$LOG_FILE"
    else
        echo "Warning: $PVTU not found in $SRC_DIR"
    fi

    # Copy 16 .vtu part files
    for IDX in {0..15}; do
        PART=$(printf "%s.%04d.vtu" "$VTU_BASE" "$IDX")
        if [ -f "$SRC_DIR/$PART" ]; then
            cp "$SRC_DIR/$PART" "$DEST_SUBDIR/"
            echo "$DEST_SUBDIR/$PART" >> "$LOG_FILE"
        else
            echo "Warning: $PART not found in $SRC_DIR"
        fi
    done
done

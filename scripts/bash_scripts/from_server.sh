#!/usr/bin/env bash

set -euo pipefail

SERVER=${1:-hive}
STEP="${2:-}"
    
EXCLUDES=(
    "--exclude=*restart*"
)

source "${HaMaGeoLib_DIR}/scripts/bash_scripts/find_project_index.sh" "$SERVER"

REMOTE_PATH="${SERVER}:${selected_remote_folder}/${relative_case_path}/*"

if [[ -n "$STEP" ]]; then

    FILE_PATTERN=$(printf "solution-%05d.*" "$STEP")

    echo "Selective sync:"
    echo "  ${FILE_PATTERN}"
    echo "rsync -avu  --include='*/' --include=output/solution/${FILE_PATTERN} --exclude='*' ${REMOTE_PATH} ."

    rsync -avu \
        --include='*/' \
        --include="output/solution/${FILE_PATTERN}" \
        --exclude='*' \
        "${REMOTE_PATH}" \
        .

else


    echo "Full sync"
    echo "rsync -avu ${EXCLUDES[*]} ${REMOTE_PATH} ."

    rsync -avu \
        "${EXCLUDES[@]}" \
        "${REMOTE_PATH}" \
        .

fi
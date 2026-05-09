#!/usr/bin/env bash
# todo_sync

set -euo pipefail

SERVER=${1:-hive}

source "${HaMaGeoLib_DIR}/scripts/bash_scripts/find_project_index.sh" "$SERVER"

EXCLUDES=(
    "--exclude=*restart*"
)

REMOTE_PATH="${SERVER}:${selected_remote_folder}/${relative_case_path}/*"

echo "Running:"
echo "rsync -avu ${EXCLUDES[*]} ${REMOTE_PATH} ."

rsync -avu \
    "${EXCLUDES[@]}" \
    "${REMOTE_PATH}" \
    .
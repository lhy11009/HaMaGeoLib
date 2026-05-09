#!/usr/bin/env bash

set -euo pipefail

SERVER=${1:-hive}

source "${HaMaGeoLib_DIR}/scripts/bash_scripts/find_project_index.sh" "$SERVER"

# first create parent folder
echo "ssh $SERVER mkdir -p ${selected_remote_folder}/${relative_case_path}"
ssh $SERVER "mkdir -p ${selected_remote_folder}/${relative_case_path}"

INCLUDES=(
    "--include=*.prm"
    "--include=*.wb"
)

REMOTE_PATH="${SERVER}:${selected_remote_folder}/${relative_case_path}/"

# run command to sync from local to remote folder
echo "rsync -avu ${INCLUDES[*]} --exclude=* ./* $REMOTE_PATH"

rsync -avu \
    "${INCLUDES[@]}" \
    --exclude=* \
    ./* \
    "${REMOTE_PATH}"

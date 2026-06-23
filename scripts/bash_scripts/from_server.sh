#!/usr/bin/env bash

set -euo pipefail

INCLUDE_RESTART=false
INCLUDE_PARTICLES=false
POSITIONAL=()

# parse the command line options
for arg in "$@"; do
    case "$arg" in
        -r)
            INCLUDE_RESTART=true
            ;;
        -p)
            INCLUDE_PARTICLES=true
            ;;
        *)
            POSITIONAL+=("$arg")
            ;;
    esac
done


# read server and step
SERVER=${POSITIONAL[0]:-hive}
STEP=${POSITIONAL[1]:-}
    
# exclude patterns from sync
# Only exclude particles if -p was NOT specified
EXCLUDES=()

if ! $INCLUDE_RESTART; then
    EXCLUDES+=("--exclude=*restart*")
fi

if ! $INCLUDE_PARTICLES; then
    EXCLUDES+=("--exclude=*particle*")
fi


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
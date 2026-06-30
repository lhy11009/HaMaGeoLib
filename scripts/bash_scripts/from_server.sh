#!/usr/bin/env bash

set -euo pipefail

INCLUDE_RESTART=false
INCLUDE_ONLY_RESTART=false
INCLUDE_PARTICLES=false
POSITIONAL=()

# parse the command line options
# Parse command line options
# -l: local path for keeping the data
#   note this is the parent directory
#   a case directory will be created 
#   in it before syncing
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r)
            INCLUDE_RESTART=true
            shift
            ;;
        -R)
            INCLUDE_ONLY_RESTART=true
            shift
            ;;
        -p)
            INCLUDE_PARTICLES=true
            shift
            ;;
        -l)
            LOCAL_DIR="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
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

# Derive the remote path
REMOTE_PATH="${SERVER}:${selected_remote_folder}/${relative_case_path}/*"

# todo_restart
# Derive the local path
# By default, rsync to the current directory.
# If a local directory is assigned, then first make a case direction under it
# and then sync to it.
LOCAL_PATH="."

if [[ -n "$LOCAL_DIR" ]]; then
    LOCAL_PATH="$LOCAL_DIR/${relative_case_path}/"
    mkdir -p "${LOCAL_PATH}"
fi

# Sync data
# In case the $INCLUDE_ONLY_RESTART is true, we only sync the restart file.
# This option is used if we want to keep restart file in a separate location
# Else if a $STEP is given, then only a certain step is synced
# Otherwise the case is synced as a whole
if [[ $INCLUDE_ONLY_RESTART ]]; then

    echo "rsync -avu  --include='*/' --include='output/restart/*' --include='output/restart/*/*' --exclude='*' ${REMOTE_PATH} ${LOCAL_PATH}"

    eval "rsync -avu  --include='*/' --include='output/restart/*' --include='output/restart/*/*' --exclude='*' ${REMOTE_PATH} ${LOCAL_PATH}"

elif [[ -n "$STEP" ]]; then

    FILE_PATTERN=$(printf "solution-%05d.*" "$STEP")

    echo "Selective sync:"
    echo "  ${FILE_PATTERN}"
    echo "rsync -avu  --include='*/' --include=output/solution/${FILE_PATTERN} --exclude='*' ${REMOTE_PATH} ${LOCAL_PATH}"

    eval "rsync -avu  --include='*/' --include=output/solution/${FILE_PATTERN} --exclude='*' ${REMOTE_PATH} ${LOCAL_PATH}"

else

    echo "Full sync"
    echo "rsync -avu ${EXCLUDES[*]} ${REMOTE_PATH} ${LOCAL_PATH}"
    
    eval "rsync -avu ${EXCLUDES[*]} ${REMOTE_PATH} ${LOCAL_PATH}"

fi
#!/usr/bin/env bash

source "${HaMaGeoLib_DIR}/scripts/bash_scripts/find_project_index.sh"

# run command to sync from local to remote folder
echo "rsync -avu --include=*.prm --include=*.wb --exclude=* ./* hive:${selected_folder_hive}/${relative_case_path}/"
eval "rsync -avu --include=*.prm --include=*.wb --exclude=* ./* hive:${selected_folder_hive}/${relative_case_path}/"


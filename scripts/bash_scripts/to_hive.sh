#!/usr/bin/env bash

source "${HaMaGeoLib_DIR}/scripts/bash_scripts/find_project_index.sh"

# first create parent folder
echo "ssh hive mkdir -p ${selected_folder_hive}/${relative_case_path}"
ssh hive "mkdir -p ${selected_folder_hive}/${relative_case_path}"

# run command to sync from local to remote folder
echo "rsync -avu --include=*.prm --include=*.wb --exclude=* ./* hive:${selected_folder_hive}/${relative_case_path}/"
eval "rsync -avu --include=*.prm --include=*.wb --exclude=* ./* hive:${selected_folder_hive}/${relative_case_path}/"


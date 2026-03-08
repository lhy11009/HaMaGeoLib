#!/usr/bin/env bash

source "${HaMaGeoLib_DIR}/scripts/bash_scripts/find_project_index.sh"

# run command to sync from remote folder to local
echo "rsync -avu --exclude=*restart* --exclude=*particle* hive:"${selected_folder_hive}/${relative_case_path}/*" ."
eval "rsync -avu --exclude=*restart* --exclude=*particle* hive:"${selected_folder_hive}/${relative_case_path}/*" ."


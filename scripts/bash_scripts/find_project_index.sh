#!/usr/bin/env bash
# This script finds the index of project that contains the current case folder

find_project_index () {

    local abs_path
    abs_path=$(pwd)

    # ---- find_project_index ----
    local i
    IFS=":" read -ra PROJECT_DIRS <<< "$PROJECT"
    IFS=":" read -ra PROJECT_HIVE_DIRS <<< "$PROJECT_HIVE"
    i=$("${HaMaGeoLib_DIR}/scripts/bash_scripts/find_parent_dir_index.sh" \
        "$abs_path" "${PROJECT_DIRS[@]}")

    if [[ -z "$i" ]]; then
        echo "Error: find_parent_dir_index.sh returned empty result."
        echo "Path checked: $abs_path"
        echo "PROJECT directories:"
        printf '  %s\n' "${PROJECT}"
        return 2
    fi
    if (( i >= 0 )); then
        export selected_folder="${PROJECT_DIRS[$i]}"
        export selected_folder_hive="${PROJECT_HIVE_DIRS[$i]}"
        export relative_case_path="${abs_path#"$selected_folder"/}"
    else
        echo "No matching folder found"
        echo "Path: $abs_path"
        echo "Projects: ${PROJECT}"
        return 1
    fi

    # ---- sync from hive ----
}


find_project_index

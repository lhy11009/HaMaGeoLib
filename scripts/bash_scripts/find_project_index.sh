#!/usr/bin/env bash
# This script finds the index of project that contains the current case folder

find_project_index () {

    local server="$1"

    : "${PROJECT:?PROJECT is not set}"

    local remote_var="PROJECT_REMOTE_${server}"

    local remote_projects="${!remote_var:-}"

    if [[ -z "$remote_projects" ]]; then
        echo "Error: remote project variable not found:"
        echo "  $remote_var"
        return 4
    fi

    local abs_path
    abs_path=$(pwd)

    IFS=":" read -ra PROJECT_DIRS <<< "$PROJECT"
    IFS=":" read -ra REMOTE_DIRS <<< "$remote_projects"

    if (( ${#PROJECT_DIRS[@]} != ${#REMOTE_DIRS[@]} )); then
        echo "Error: PROJECT and ${remote_var} length mismatch."
        return 3
    fi

    local i

    i=$(
        "${HaMaGeoLib_DIR}/scripts/bash_scripts/find_parent_dir_index.sh" \
        "$abs_path" \
        "${PROJECT_DIRS[@]}"
    )

    if [[ -z "$i" ]]; then
        echo "Error: find_parent_dir_index.sh returned empty result."
        return 2
    fi

    if (( i >= 0 )); then

        export selected_folder="${PROJECT_DIRS[$i]}"

        export selected_remote_folder="${REMOTE_DIRS[$i]}"

        export relative_case_path="${abs_path#"$selected_folder"/}"

        # assert these are not empty
        : "${selected_remote_folder:?selected_remote_folder is empty}"
        : "${relative_case_path:?relative_case_path is empty}"

    else

        echo "No matching folder found"
        echo "Path: $abs_path"

        return 1
    fi
}

find_project_index "$1"

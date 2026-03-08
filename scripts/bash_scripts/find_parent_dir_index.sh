#!/usr/bin/env bash

# Usage:
#   find_parent_index "/abs/path/to/file" "/folder1" "/folder2" "/folder3"

find_parent_index () {

    local target_path="$1"
    shift

    local folders=("$@")

    for i in "${!folders[@]}"; do
        folder="${folders[$i]}"

        # normalize paths (remove trailing slash)
        folder="${folder%/}"

        if [[ "$target_path" == "$folder"* ]]; then
            echo "$i"
            return 0
        fi
    done

    echo "-1"
    return 1
}


# Usage from command line:
#   inputs are the target path and a list of potential top level paths.
#   The typical purpose of this is to select a project that contains the
# current case folder.
target="$1"
shift

find_parent_index "$target" "$@"
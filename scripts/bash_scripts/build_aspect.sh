#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Build an ASPECT branch and its plugins.

Usage:
    build_aspect.sh <branch> [options]

Options:
    --mode <mode>       Build mode: debugrelease (default), debug, or release.
    --tag <tag>         Append a user-provided tag to the build directory.
    --date-tag          Append today's date in yymmdd format.
    --configure         Remove the build directory and configure from scratch.
    --plugin <name>     Build only this plugin. May be specified more than once.
    --no-plugins        Do not build plugins.
    -j, --jobs <count>  Number of parallel build jobs (default: 1).
    -h, --help          Show this help message.

The ASPECT source directory must be set in ASPECT_SOURCE_DIR. Build directories
are named build_<branch> or build_<branch>_<tag> beneath ASPECT_SOURCE_DIR.
EOF
}

die() {
    printf 'build_aspect.sh: %s\n' "$*" >&2
    return 2
}

compute_build_directory() {
    local requested_branch="$1"
    local requested_tag="${2:-}"
    local directory_branch

    : "${ASPECT_SOURCE_DIR:?ASPECT_SOURCE_DIR must be set}"
    directory_branch="${requested_branch//\//_}"

    if [[ -n "$requested_tag" ]]; then
        printf '%s/build_%s_%s\n' "$ASPECT_SOURCE_DIR" "$directory_branch" "$requested_tag"
    else
        printf '%s/build_%s\n' "$ASPECT_SOURCE_DIR" "$directory_branch"
    fi
}

parse_arguments() {
    branch=""
    mode="debugrelease"
    tag=""
    configure=0
    build_plugins=1
    jobs=1
    plugins=()
    local manual_tag=0
    local date_tag=0

    while (($#)); do
        case "$1" in
            --mode)
                (($# >= 2)) || die '--mode requires an argument'
                mode="$2"
                shift 2
                ;;
            --tag)
                (($# >= 2)) || die '--tag requires an argument'
                tag="$2"
                manual_tag=1
                shift 2
                ;;
            --date-tag)
                date_tag=1
                shift
                ;;
            --configure)
                configure=1
                shift
                ;;
            --plugin)
                (($# >= 2)) || die '--plugin requires an argument'
                plugins+=("$2")
                shift 2
                ;;
            --no-plugins)
                build_plugins=0
                shift
                ;;
            -j|--jobs)
                (($# >= 2)) || die "$1 requires an argument"
                jobs="$2"
                shift 2
                ;;
            -h|--help)
                usage
                return 64
                ;;
            --*)
                die "unknown option: $1"
                ;;
            *)
                [[ -z "$branch" ]] || die "unexpected argument: $1"
                branch="$1"
                shift
                ;;
        esac
    done

    [[ -n "$branch" ]] || die 'an ASPECT branch is required'
    [[ "$mode" == debugrelease || "$mode" == debug || "$mode" == release ]] || \
        die "invalid mode: $mode"
    [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die 'jobs must be a positive integer'
    ((manual_tag == 0 || date_tag == 0)) || die '--tag and --date-tag cannot be used together'
    ((build_plugins == 1 || ${#plugins[@]} == 0)) || \
        die '--plugin and --no-plugins cannot be used together'

    if ((date_tag)); then
        tag="$(date +%y%m%d)"
    fi
    if [[ -n "$tag" && ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
        die 'tag may contain only letters, numbers, dots, underscores, and hyphens'
    fi
    for plugin in "${plugins[@]}"; do
        if [[ ! "$plugin" =~ ^[A-Za-z0-9._-]+$ || "$plugin" == . || "$plugin" == .. ]]; then
            die 'plugin names may contain only letters, numbers, dots, underscores, and hyphens'
        fi
    done
}

discover_plugins() {
    local plugins_directory="$ASPECT_SOURCE_DIR/plugins"
    local plugin_directory

    [[ -d "$plugins_directory" ]] || return 0
    for plugin_directory in "$plugins_directory"/*; do
        [[ -d "$plugin_directory" ]] || continue
        basename "$plugin_directory"
    done
}

prepare_build_directory() {
    local build_directory="$1"

    [[ "$build_directory" == "$ASPECT_SOURCE_DIR"/build_* ]] || \
        die "refusing to prepare unexpected build directory: $build_directory"
    if ((configure)); then
        rm -rf -- "$build_directory"
    fi
    mkdir -p -- "$build_directory"
}

build_plugin() {
    local build_directory="$1"
    local plugin="$2"
    local plugin_source="$ASPECT_SOURCE_DIR/plugins/$plugin"
    local plugin_target="$build_directory/$plugin"

    [[ -d "$plugin_source" ]] || die "plugin directory does not exist: $plugin_source"

    rm -rf -- "$plugin_target"
    cp -R -- "$plugin_source" "$plugin_target"
    cmake -S "$plugin_target" -B "$plugin_target" -DAspect_DIR="$build_directory"
    cmake --build "$plugin_target" --parallel "$jobs"
}

build_aspect() {
    local build_directory
    local cmake_arguments=()
    local plugin

    : "${ASPECT_SOURCE_DIR:?ASPECT_SOURCE_DIR must be set}"
    [[ -d "$ASPECT_SOURCE_DIR/.git" ]] || die "not an ASPECT repository: $ASPECT_SOURCE_DIR"

    build_directory="$(compute_build_directory "$branch" "$tag")"
    git -C "$ASPECT_SOURCE_DIR" switch "$branch"
    prepare_build_directory "$build_directory"

    if [[ -n "${WORLD_BUILDER_SOURCE_DIR:-}" ]]; then
        cmake_arguments+=("-DWORLD_BUILDER_SOURCE_DIR=$WORLD_BUILDER_SOURCE_DIR")
    fi
    if ((configure)) || [[ ! -f "$build_directory/CMakeCache.txt" ]]; then
        cmake -S "$ASPECT_SOURCE_DIR" -B "$build_directory" "${cmake_arguments[@]}"
    fi

    cmake --build "$build_directory" --target "$mode"
    cmake --build "$build_directory" --parallel "$jobs"

    ((build_plugins)) || return 0
    if ((${#plugins[@]} == 0)); then
        mapfile -t plugins < <(discover_plugins)
    fi
    for plugin in "${plugins[@]}"; do
        build_plugin "$build_directory" "$plugin"
    done
}

main() {
    if (($# == 0)); then
        usage >&2
        return 2
    fi

    if [[ "$1" == -h || "$1" == --help ]]; then
        usage
        return 0
    fi

    parse_arguments "$@"
    build_aspect
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi

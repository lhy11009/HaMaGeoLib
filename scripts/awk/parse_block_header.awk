#!/usr/bin/awk -f

BEGIN {
    printf "#%9s %10s %10s %10s\n", "ASPECT", "deal.II", "WORLD_BUILDER", "MPI"
    version = "nan"
    deal_ii_version = "nan"
    world_builder_version = "nan"
    number_of_mpi = "nan"
}

{
    if ($0 ~ /\. version/) {
        version = $4
    }
    else if ($0 ~ /\. using deal.II/) {
        deal_ii_version = $5
    }
    else if ($0 ~ /\. using Geodynamic World Builder/) {
        world_builder_version = $7
    }
    else if ($0 ~ /\. running with/) {
        number_of_mpi = $5
        printf "%10s %10s %10s %10s\n", version, deal_ii_version, world_builder_version, number_of_mpi
        exit
    }
}

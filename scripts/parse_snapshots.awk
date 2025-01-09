#!/usr/bin/awk -f

################################################################################
# AWK Script to Parse Snapshots from an ASPECT Log File
#
# Purpose:
# This script processes an ASPECT log file and extracts information about
# snapshots. It generates a list of steps where snapshots are created,
# including details such as the timestep number, simulation time, and total
# wall clock time.
#
# Example Usage:
# eval "awk -f ${ASPECT_LAB_DIR}/bash_scripts/awk_states/parse_snapshot \
#       /home/lochy/ASPECT_PROJECT/TwoDSubduction/non_linear32/eba1_MRf12_iter20_DET/output/log.txt > test_output"
#
# Input:
# The script expects a path to an ASPECT log file.
#
# Output:
# A formatted output with the following columns:
#   1. Time step number
#   2. Time
#   3. Wall Clock (s)
#
# Header:
# The script starts by printing a header explaining the output format.
################################################################################

# Print the header for the output file
BEGIN {
  print "# 1: Time step number\n# 2: Time\n# 3: Wall Clock (s)"
}

# Process each line of the log file
{
  # Extract the total wall clock time when encountering a relevant line
  if (lastLine ~ /\*\*\* Timestep/ && $0 ~ /^\| Total wallclock/) {
    len = length($9)
    wallClock = substr($9, 0, len-1)  # Remove the trailing character from the wall clock value
  }
  # Store information about the timestep and simulation time
  else if ($0 ~ /\*\*\* Timestep/) {
    lastLine = $0
    len = length($3)
    timeStep = substr($3, 0, len-1)  # Remove the trailing character from the timestep value
    time = substr($4, 3)            # Extract the simulation time, skipping unnecessary characters
  }
  # Output information when a snapshot is created
  else if ($0 ~ /\*\*\* Snapshot created/) {
    printf "%s %12s %10s\n", timeStep, time, wallClock
  }
}

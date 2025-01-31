#!/usr/bin/awk -f

################################################################################
# Script: parse_block_output.awk
# Author: Haoyuan Li
# License: MIT
#
# Description:
#   This script extracts runtime information, including time step number, 
#   simulation time, and total wall clock time, from an ASPECT log file. 
#   The extracted information is formatted into a structured output.
#
# Example Usage:
#   eval "awk -f ${ASPECT_LAB_DIR}/bash_scripts/awk_states/parse_block_output.awk ${log_file} > ${ofile}"
#
# Output format:
#   The script outputs three columns:
#     1. Time step number
#     2. Simulation time
#     3. Wall clock time (in seconds)
################################################################################

# Print the header for the extracted data
BEGIN {
  print "# 1: Time step number\n# 2: Time\n# 3: Wall Clock (s)"
}

{
  # Check if the last line contained timestep info and the current line contains total wall clock time
  if (lastLine ~ /\*\*\* Timestep/ && $0 ~ /^\| Total wallclock/) {
    len = length($9)                 # Get the length of the wall clock time string
    wallClock = substr($9, 0, len-1)  # Extract the numerical value by removing the last character
    printf "%s %12s %10s\n", timeStep, time, wallClock  # Print the extracted values
  }
  # If the line contains timestep information, extract relevant values
  else if ($0 ~ /\*\*\* Timestep/) {
    lastLine = $0                     # Store the current line for later reference
    len = length($3)                   # Get the length of the timestep string
    timeStep = substr($3, 0, len-1)     # Extract timestep number by removing the last character
    time = substr($4, 3)                # Extract the time value, skipping first two characters
  }
}

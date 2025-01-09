#!/usr/bin/awk -f

################################################################################
# Script Purpose:
# Extract runtime information about the Newton solver from an ASPECT log file.
# Specifically, this script retrieves:
#   - Time step number
#   - Number of "cheap" and "expensive" Stokes solver iterations
#   - Nonlinear iteration index
#   - Relative nonlinear residual
#   - Norm of the right-hand side (RHS)
#   - Newton derivative scaling factor
#
# Example Usage:
#   eval "awk -f ${ASPECT_LAB_DIR}/bash_scripts/awk_states/parse_block_newton \
#        ${log_file} > ${ofile}"
################################################################################

BEGIN {
  # Print header for the output table
  print "# 1: Time step number"
  print "# 2: Number of Cheap Stokes iterations"
  print "# 3: Number of Expensive Stokes iterations"
  print "# 4: Index of nonlinear iteration"
  print "# 5: Relative nonlinear residual"
  print "# 6: Norms of the RHS"
  print "# 7: Newton Derivative Scaling Factor"
}

# Process lines related to solving the Stokes system
{
  if (lastLine ~ /\*\*\* Timestep/ && $0 ~ /Solving Stokes system/) {
    # Parse the number of "cheap" and "expensive" Stokes solver iterations
    split($4, array, "+")
    stokes_iteration_cheap = array[1]
    stokes_iteration_expensive = array[2]
    
    # Handle cases where the "expensive" count is missing
    if (length(stokes_iteration_expensive) == 0) {
      stokes_iteration_expensive = "-1"
    }
  }
  # Process lines with nonlinear residual data
  else if (lastLine ~ /\*\*\* Timestep/ && $0 ~ /Relative nonlinear residual/) {
    # Extract nonlinear iteration index and residual value
    len = length($10)
    nonlinear_iteration = substr($10, 0, len-1)
    len = length($11)
    residual = substr($11, 0, len-1)
    
    # Extract and clean up the norm of the RHS
    norm_rhs = $16
    sub(",", "", norm_rhs)

    # Extract the Newton derivative scaling factor, default to "0" if missing
    if (length($18) == 0) {
      scaling_factor = "0"
    } else {
      scaling_factor = $18
    }

    # Print the extracted values in a structured format
    printf "%s %10s %10s %15s %15s %15s %15s\n", \
           timeStep, stokes_iteration_cheap, stokes_iteration_expensive, \
           nonlinear_iteration, residual, norm_rhs, scaling_factor
  }
  # Identify lines marking a new timestep and extract its number
  else if ($0 ~ /\*\*\* Timestep/) {
    lastLine = $0
    len = length($3)
    timeStep = substr($3, 0, len-1)
  }
}

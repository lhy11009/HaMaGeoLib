# Minimum CMake version required
cmake_minimum_required(VERSION 3.12)

# Project name and version
project(MetastableProject VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Create the executable for tests
add_executable(metastable_test tests/metastable_test.cpp src/metastable.cpp src/IRK4Solver.cpp)
add_executable(IRK4Solver_test tests/IRK4Solver_test.cpp src/IRK4Solver.cpp)

# Create the executables
set(SOURCES_METASTABLE_DIAGRAM
    src/metastable.cpp
    src/IRK4Solver.cpp
    src/metastable_generate_diagram.cpp
)

# Create the executable
add_executable(metastable_diagram ${SOURCES_METASTABLE_DIAGRAM})

# Enable debugging with gdb
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -g -O0")
    add_definitions(-DDEBUG)  # Define DEBUG macro
endif()

# Generate compile_commands.json, for better support with IntelliSense support in VS Code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
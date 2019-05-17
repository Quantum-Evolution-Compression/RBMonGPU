cmake_minimum_required(VERSION 3.8)
project(pyRBMonGPU)

# set helper pathes to find libraries and packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/lib")

set(LIBS ${LIBS} "$ENV{HOME}/.local/lib")

# Language Flags
# ==============

# enforce C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)


# Project paths
# =============

set(project_include_dir "${CMAKE_CURRENT_LIST_DIR}/../include")
set(project_source_dir "${CMAKE_CURRENT_LIST_DIR}/../source")

# Project's includes
# ==================

set(CXX_INCLUDES ${CXX_INCLUDES} ${project_include_dir})


# Find Cuda
# =========

find_package(CUDA REQUIRED)
set(CXX_INCLUDES ${CXX_INCLUDES} ${CUDA_INCLUDE_DIRS})
# set(LIBS ${LIBS} ${CUDA_LIBRARIES})

# Find Python
# ===========

if(DEFINED ENV{PYTHONHOME})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{PYTHONHOME}")
endif()
find_package(PythonLibs REQUIRED)
set(CXX_INCLUDES ${CXX_INCLUDES} ${PYTHON_INCLUDE_DIR})

# Find NumPy
# ==========

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake" BEFORE)

find_package(NumPy REQUIRED)
set(CXX_INCLUDES ${CXX_INCLUDES} ${NUMPY_INCLUDE_DIRS})


# Compiler flags
# ==============

# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


set(PYBIND11_CPP_STANDARD -std=c++14)
find_package(pybind11 REQUIRED)
pybind11_add_module(_pyRBMonGPU "${CMAKE_CURRENT_LIST_DIR}/pyRBMonGPU/main.cpp")

link_directories(${LIBS})

target_include_directories(_pyRBMonGPU PRIVATE ${CXX_INCLUDES})
target_link_libraries(_pyRBMonGPU PRIVATE RBMonGPU)
# target_link_libraries(_pyRBMonGPU PRIVATE ${LIBS})

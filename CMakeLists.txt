cmake_minimum_required(VERSION 3.8)
project(RBMonGPU LANGUAGES CXX CUDA)

# set helper pathes to find libraries and packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")


# Language Flags
# ==============

# enforce C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)

# Project paths
# =============

set(project_include_dir "${CMAKE_CURRENT_LIST_DIR}/include")
set(project_source_dir "${CMAKE_CURRENT_LIST_DIR}/source")
set(project_bin_dir "${CMAKE_CURRENT_LIST_DIR}/bin")

# Find Cuda
# =========

find_package(CUDA REQUIRED)
set(CXX_INCLUDES ${CXX_INCLUDES} ${CUDA_INCLUDE_DIRS})
set(LIBS ${LIBS} ${CUDA_LIBRARIES})

# Find Python
# ===========

if(DEFINED ENV{PYTHONHOME})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{PYTHONHOME}")
endif()
find_package(PythonLibs REQUIRED)
set(CXX_INCLUDES ${CXX_INCLUDES} ${PYTHON_INCLUDE_DIR})

# Find NumPy
# ==========

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" BEFORE)

find_package(NumPy REQUIRED)
set(CXX_INCLUDES ${CXX_INCLUDES} ${NUMPY_INCLUDE_DIRS})

# Project's includes
# ==================

set(CUDA_INCLUDES ${CUDA_INCLUDES} ${project_include_dir})
set(CXX_INCLUDES ${CXX_INCLUDES} ${project_include_dir})

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

# NVCC flags
# ==========

if(NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-arch=sm_70 --default-stream per-thread --expt-extended-lambda" CACHE STRING "nvcc flags" FORCE)
endif()

if(TIMING)
    add_definitions(-DTIMING)
endif()

if(MAX_SPINS)
    add_compile_definitions(MAX_SPINS=${MAX_SPINS})
endif()

# Set target
# ==========

file(GLOB_RECURSE CUDASRCFILES "${project_source_dir}/*")
add_library(RBMonGPU SHARED ${CUDASRCFILES})

# User options
# ==============

option(ENABLE_MONTE_CARLO "enable monte-carlo" ON)
if (ENABLE_MONTE_CARLO)
  target_compile_definitions(RBMonGPU PRIVATE ENABLE_MONTE_CARLO)
endif()

option(ENABLE_EXACT_SUMMATION "enable exact summation" ON)
if (ENABLE_EXACT_SUMMATION)
  target_compile_definitions(RBMonGPU PRIVATE ENABLE_EXACT_SUMMATION)
endif()

option(ENABLE_PSI "enable Psi" ON)
if (ENABLE_PSI)
  target_compile_definitions(RBMonGPU PRIVATE ENABLE_PSI)
endif()

option(ENABLE_PSI_DEEP "enable PsiDeep" ON)
if (ENABLE_PSI_DEEP)
  target_compile_definitions(RBMonGPU PRIVATE ENABLE_PSI_DEEP)
endif()

option(ENABLE_PSI_DEEP_MIN "enable PsiDeepMin" ON)
if (ENABLE_PSI_DEEP_MIN)
  target_compile_definitions(RBMonGPU PRIVATE ENABLE_PSI_DEEP_MIN)
endif()

option(ENABLE_PSI_HAMILTONIAN "enable PsiHamiltonian" ON)
if (ENABLE_PSI_HAMILTONIAN)
  target_compile_definitions(RBMonGPU PRIVATE ENABLE_PSI_HAMILTONIAN)
endif()

option(ENABLE_PSI_CLASSICAL "enable PsiClassical" ON)
if (ENABLE_PSI_CLASSICAL)
  target_compile_definitions(RBMonGPU PRIVATE ENABLE_PSI_CLASSICAL)
endif()

option(ENABLE_PSI_PAIR "enable PsiPair" ON)
if (ENABLE_PSI_PAIR)
  target_compile_definitions(RBMonGPU PRIVATE ENABLE_PSI_PAIR)
endif()

# Compile and link RBMonGPU
# =========================


# file(GLOB_RECURSE SRCFILES "${project_source_dir}/*.cpp")


# add_executable(test ${project_source_dir}/main.cpp)

target_include_directories(RBMonGPU PRIVATE ${CUDA_INCLUDES})
target_include_directories(RBMonGPU PRIVATE ${CXX_INCLUDES})
# target_include_directories(test PRIVATE ${CXX_INCLUDES})

target_link_libraries(RBMonGPU ${LIBS})
# target_link_libraries(test ${LIBS})
# target_link_libraries(test RBMonGPU)

# add_dependencies(test RBMonGPU)

# Installation
# ============

install(TARGETS RBMonGPU LIBRARY DESTINATION lib)
install(
	DIRECTORY "${project_include_dir}/"
	DESTINATION include/RBMonGPU
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)


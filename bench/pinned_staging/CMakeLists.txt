# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Standalone microbenchmark: pageable -> pinned ring -> device double buffering
# versus a fully pinned source. Not part of the extension build; see README.md.

cmake_minimum_required(VERSION 3.30.4)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(pinned_staging_bench LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

add_executable(pinned_staging_bench pinned_staging_bench.cu)

set_target_properties(
  pinned_staging_bench
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON)

# NVTX v3 is header-only (loads the injection library via dlopen). In a conda /
# pixi environment it is installed under $CONDA_PREFIX/include rather than the
# CUDA target include directory nvcc searches, so locate it explicitly.
find_path(
  NVTX3_INCLUDE_DIR nvtx3/nvToolsExt.h
  HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} $ENV{CONDA_PREFIX}/include
        $ENV{CUDA_HOME}/include /usr/local/cuda/include
  DOC "Directory containing nvtx3/nvToolsExt.h")
if(NOT NVTX3_INCLUDE_DIR)
  message(FATAL_ERROR "nvtx3/nvToolsExt.h not found; set NVTX3_INCLUDE_DIR")
endif()
target_include_directories(pinned_staging_bench PRIVATE ${NVTX3_INCLUDE_DIR})
target_link_libraries(pinned_staging_bench PRIVATE Threads::Threads
                                                   ${CMAKE_DL_LIBS})

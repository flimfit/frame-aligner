# ########################################################################
# Copyright 2013 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################


# Locate the FFTW3 (http://www.FFTW3.org/) Framework.
#
# Defines the following variables:
#
#   FFTW3_FOUND - Found the FFTW3 framework
#   FFTW3_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   FFTW3_LIBRARIES - libFFTW3
#
# Accepts the following variables as input:
#
#   FFTW3_ROOT - (as a CMake or environment variable)
#                The root directory of the FFTW3 install prefix
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether 
#               findFFTW3 should search for 64bit or 32bit libs
#-----------------------------------------------
# Example Usage:
#
#    find_package(FFTW3 REQUIRED)
#    include_directories(${FFTW3_INCLUDE_DIRS})
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${FFTW3_LIBRARIES})
#
#-----------------------------------------------

find_path(FFTW3_INCLUDE_DIRS
NAMES FFTW3.h
HINTS
    ${FFTW3_ROOT}/include
    ${FFTW3_ROOT}/api
    ${FFTW3_ROOT}
    $ENV{FFTW3_ROOT}/include
    $ENV{FFTW3_ROOT}/api
    ENV FFTW3_ROOT
PATHS
    /usr/include
    /usr/local/include
)
mark_as_advanced( FFTW3_INCLUDE_DIRS )

find_library( FFTW3_DOUBLE_PRECISION_LIBRARIES
NAMES FFTW3 libFFTW3-3
HINTS
    ${FFTW3_ROOT}/lib
    ${FFTW3_ROOT}/.libs
    ${FFTW3_ROOT}
    $ENV{FFTW3_ROOT}/lib
    $ENV{FFTW3_ROOT}/.libs
    ENV FFTW3_ROOT
PATHS
    /usr/lib
    /usr/local/lib
DOC "FFTW3 dynamic library"
)
mark_as_advanced( FFTW3_DOUBLE_PRECISION_LIBRARIES )

set( FFTW3_LIBRARIES ${FFTW3_DOUBLE_PRECISION_LIBRARIES} )
mark_as_advanced( FFTW3_LIBRARIES )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( FFTW3 DEFAULT_MSG FFTW3_LIBRARIES FFTW3_INCLUDE_DIRS )

if( NOT FFTW3_FOUND )
message( STATUS "FindFFTW3 looked for double precision libraries named: FFTW3 or libFFTW3-3" )
endif()

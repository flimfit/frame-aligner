#  DLIB_INCLUDE_DIRS - where to find dlib
#  DLIB_LIBRARIES    - List of libraries for dlib.
#  DLIB_FOUND        - True if dlib found.

include(LibFindMacros)

find_path(dlib_INCLUDE_DIR libxml/xpath.h
          HINTS ${dlib_INCLUDEDIR} ${dlib_INCLUDE_DIRS}
          PATH_SUFFIXES dlib )

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(dlib_PKGCONF dlib)

# Include dir
find_path(dlib_INCLUDE_DIRS dlib HINTS ${dlib_PKGCONF_INCLUDE_DIRS} $ENV{DLIB_ROOT})

find_library(dlib_LIB NAMES dlib HINTS ${dlib_PKGCONF_LIBRARY_DIRS} $ENV{DLIB_ROOT})
find_library(dlib_LIB_DEBUG NAMES dlibd HINTS ${dlib_PKGCONF_LIBRARY_DIRS} $ENV{DLIB_ROOT})

if(NOT dlib_LIB_DEBUG)
   set(dlib_LIB_DEBUG ${dlib_LIB})
endif()

set(dlib_LIBRARIES optimized ${dlib_LIB} debug ${dlib_LIB_DEBUG})
message(${dlib_LIBRARIES})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set dlib_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(dlib  DEFAULT_MSG
                                  dlib_LIB dlib_INCLUDE_DIRS)


mark_as_advanced(dlib_INCLUDE_DIRS dlib_LIBRARIES)

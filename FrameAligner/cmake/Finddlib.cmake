#  DLIB_INCLUDE_DIRS - where to find dlib
#  DLIB_LIBRARIES    - List of libraries for dlib.
#  DLIB_FOUND        - True if dlib found.


find_package(PkgConfig)

find_path(dlib_INCLUDE_DIR libxml/xpath.h
          HINTS ${dlib_INCLUDEDIR} ${dlib_INCLUDE_DIRS}
          PATH_SUFFIXES dlib )

find_library(dlib_LIBRARY NAMES dlib
             HINTS ${dlib_LIBDIR} ${dlib_LIBRARY_DIRS} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set dlib_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(dlib  DEFAULT_MSG
                                  dlib_LIBRARY dlib_INCLUDE_DIR)

mark_as_advanced(dlib_INCLUDE_DIR dlib_LIBRARY )

set(dlib_LIBRARIES ${dlib_LIBRARY} )
set(dlib_INCLUDE_DIRS ${dlib_INCLUDE_DIR} )


#include(LibFindMacros)

# Use pkg-config to get hints about paths
#libfind_pkg_check_modules(dlib_PKGCONF dlib)

# Include dir
#find_path(dlib_INCLUDE_DIRS dlib HINTS ${dlib_PKGCONF_INCLUDE_DIRS} $ENV{DLIB_ROOT})

#find_library(dlib_LIB NAMES dlib HINTS ${dlib_PKGCONF_LIBRARY_DIRS} $ENV{DLIB_ROOT}/lib)
#find_library(dlib_LIB_DEBUG NAMES dlib_d HINTS ${dlib_PKGCONF_LIBRARY_DIRS} $ENV{DLIB_ROOT}/lib)
#set(dlib_LIBRARIES optimized ${DLIB_LIB} debug ${DLIB_LIB_DEBUG})

#message("ROOT: ${dlib_PKGCONF_LIBRARY_DIRS}")
#message("DLIB: ${dlib_INCLUDE_DIRS}, ${dlib_LIBRARIES}")

#set(dlib_PROCESS_INCLUDES dlib_INCLUDE_DIR)
#set(dlib_PROCESS_LIBS dlib_LIBRARIES)
#libfind_process(dlib)

#mark_as_advanced(dlib_INCLUDE_DIRS dlib_LIBRARIES)

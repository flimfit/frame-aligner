cmake_minimum_required(VERSION 3.7)
set(CMAKE_CXX_STANDARD 14)

include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.18.43.tar.gz"
    SHA1 "d2c8c42cd07f7cefe18fd9a9b9c13114b1a15a27"
    LOCAL
)


project(FrameAligner)

hunter_add_package(OpenCV)
hunter_add_package(dlib)

option(GENERATE_MEX "Generate Frame Aligner Mex file" ON)

if(MSVC)
   set(PLATFORM_FLAGS_ALL "/Oi /fp:fast")
   set(PLATFORM_FLAGS_RELEASE "/Ox /Ot /Ob2")
   set(PLATFORM_FLAGS_RELWITHDEBINFO "/Ox /Ot /Ob2")
   set(PLATFORM_FLAGS_DEBUG "/ZI")
   add_definitions(-D_CRT_SECURE_NO_WARNINGS)
   add_definitions(-D_MSVC)
endif(MSVC)

if(UNIX)
   set(PLATFORM_FLAGS_ALL "-msse3 -std=c++11")
   set(PLATFORM_FLAGS_RELEASE "-O3 -ffast-math -ftree-vectorize")
   set(PLATFORM_FLAGS_DEBUG "")
endif(UNIX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PLATFORM_FLAGS_ALL}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${PLATFORM_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${PLATFORM_FLAGS_DEBUG}")

set(FrameAlignerMEX_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(FrameAligner)


if(GENERATE_MEX)
   add_subdirectory(FrameAlignerMex)
endif()
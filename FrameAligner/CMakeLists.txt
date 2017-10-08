cmake_minimum_required(VERSION 3.7)
project(FrameAligner LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV 3.1 REQUIRED core imgproc highgui)
find_package(FFTW)
find_package(CUDA)
#find_package(dlib REQUIRED)

if(APPLE)
    set(EXTRA_LIBRARIES "-framework Accelerate") # for dlib
endif()

set(dlib_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/dlib)
message(dlib: ${dlib_INCLUDE_DIRS})

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

include_directories(${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(SOURCE
   AbstractFrameAligner.cpp
   RigidFrameAligner.cpp
   FrameWarpAligner.cpp
   WriteMultipageTiff.cpp
   VolumePhaseCorrelator.cpp
   Cv3dUtils.cpp
   Sobel3d.cpp
   OptimisationModel.cpp
   FrameWarper.cpp
   GpuFrameWarper.cpp
   GpuFrameWarperKernels.cu
)

set(HEADERS
   AbstractFrameAligner.h
   RigidFrameAligner.h
   FrameWarpAligner.h
   LinearInterpolation.h
   WriteMultipageTiff.h
   VolumePhaseCorrelator.h
   Cv3dUtils.h
   Sobel3d.h
   OptimisationModel.h
   FrameWarper.h
   GpuFrameWarper.h
   GpuFrameWarperKernels.h
)

set(FrameAligner_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS} PARENT_SCOPE)
set(FrameAligner_LIBRARIES ${OpenCV_LIBS} ${EXTRA_LIBRARIES} ${FFTW_LIBRARIES} dlib::dlib PARENT_SCOPE)
cuda_add_library(FrameAligner STATIC ${SOURCE} ${HEADERS})

set_target_properties(FrameAligner PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                              POSITION_INDEPENDENT_CODE ON)
                                              
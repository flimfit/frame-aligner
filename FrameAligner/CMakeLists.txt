cmake_minimum_required(VERSION 3.9)
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

list(APPEND CUDA_NVCC_FLAGS "-v")

set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_VERBOSE_BUILD ON)

include_directories(${Boost_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

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
   GpuTextureManager.h
)

set(FrameAligner_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS} PARENT_SCOPE)
set(FrameAligner_LIBRARIES ${OpenCV_LIBS} ${EXTRA_LIBRARIES} ${FFTW_LIBRARIES} dlib::dlib PARENT_SCOPE)
add_library(FrameAligner STATIC ${SOURCE} ${HEADERS})

set_target_properties(FrameAligner PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(APPLE)
# We need to add the path to the driver (libcuda.dylib) as an rpath, 
# so that the static cuda runtime can find it at runtime.
set_property(TARGET FrameAligner PROPERTY 
             BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
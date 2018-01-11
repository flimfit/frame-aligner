cmake_minimum_required(VERSION 3.9)
project(FrameAligner LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV 3.1 REQUIRED core imgproc imgcodecs videoio)
find_package(FFTW)
find_package(CUDA QUIET)

find_package(TIFF)
find_package(dlib REQUIRED)

if(APPLE)
    set(EXTRA_LIBRARIES "-framework Accelerate") # for dlib
endif()

if (CUDA_FOUND)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -arch=sm_30 -DUSE_CUDA_REALIGNMENT")
   add_definitions(-DUSE_CUDA_REALIGNMENT)
endif()

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
)

if (CUDA_FOUND)
   set(SOURCE ${SOURCE}
              GpuFrameWarper.cpp
              GpuFrameWarperKernels.cu)
   
   set(HEADERS ${HEADERS}
               GpuFrameWarper.h
               GpuFrameWarperKernels.h
               GpuTextureManager.h)
endif()

set(FrameAligner_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${FFTW_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)
set(FrameAligner_LIBRARIES ${OpenCV_LIBS} ${EXTRA_LIBRARIES} ${TIFF_LIBRARIES} ${FFTW_LIBRARIES} dlib::dlib PARENT_SCOPE)
add_library(FrameAligner STATIC ${SOURCE} ${HEADERS})

target_link_libraries(FrameAligner ${FrameAligner_LIBRARIES})

if(APPLE)
# We need to add the path to the driver (libcuda.dylib) as an rpath, 
# so that the static cuda runtime can find it at runtime.
set_property(TARGET FrameAligner PROPERTY 
             BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
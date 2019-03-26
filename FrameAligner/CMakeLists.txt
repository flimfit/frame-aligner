cmake_minimum_required(VERSION 3.9)
project(FrameAligner LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CheckLanguage)

find_package(OpenCV 3.1 REQUIRED highgui core imgproc imgcodecs videoio photo)
find_package(FFTW3 CONFIG REQUIRED)
find_package(TIFF REQUIRED)
find_package(dlib REQUIRED)
find_package(Boost COMPONENTS serialization filesystem iostreams)

set_target_properties(dlib::dlib PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND (NOT APPLE OR CMAKE_CXX_COMPILER_ID MATCHES "AppleClang") AND NOT NO_CUDA) # CUDA not supported on GCC
  enable_language(CUDA)
  find_package(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 -w")
endif()

configure_msvc_runtime("static")

#if(APPLE)
#    set(EXTRA_LIBRARIES "-framework Accelerate ${FFTW_LIBRARIES}") # for dlib
#endif()


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
   SystemMemory.cpp
   CvCache.cpp
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
   Cache.h
   Cache_impl.h
   CvCache.h
)

if (CUDA_FOUND)
   set(SOURCE ${SOURCE}
              GpuFrameWarper.cpp
              GpuFrameWarperKernels.cu)

   set(HEADERS ${HEADERS}
               GpuFrameWarper.h
               GpuFrameWarperKernels.h
               GpuTextureManager.h)

    set(EXTRA_DEFINITIONS USE_CUDA_REALIGNMENT)
endif()

add_library(FrameAligner STATIC ${SOURCE} ${HEADERS})

target_compile_definitions(FrameAligner PUBLIC ${EXTRA_DEFINITIONS})

target_link_libraries(FrameAligner PUBLIC dlib::dlib
                                          ${Boost_LIBRARIES}
                                          ${OpenCV_LIBS}
                                          ${EXTRA_LIBRARIES}
                                          ${TIFF_LIBRARIES}
                                          FFTW3::fftw3)

target_include_directories(FrameAligner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                               ${OpenCV_INCLUDE_DIRS}
                                               ${CUDA_INCLUDE_DIRS}
                                               ${FFTW_INCLUDE_DIRS})

# propagate CXX flags to CUDA host compiler
  target_compile_options(FrameAligner PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    $<$<CONFIG:DEVEL>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVEL}>
    $<$<CONFIG:STRICT>:-Xcompiler ${CMAKE_CXX_FLAGS_STRICT}>
    $<$<CONFIG:RELEASE>:-Xcompiler ${CMAKE_CXX_FLAGS_RELEASE}>
    $<$<CONFIG:DEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_DEBUG}>  
    $<$<CONFIG:HOSTDEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_HOSTDEBUG}>
    $<$<CONFIG:DEVICEDEBUG>:-Xcompiler ${CMAKE_CXX_FLAGS_DEVICEDEBUG}> 
    >)


if(CUDA_FOUND AND APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET FrameAligner PROPERTY 
                 BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

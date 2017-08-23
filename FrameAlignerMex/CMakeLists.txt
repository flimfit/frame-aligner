cmake_minimum_required(VERSION 3.7)

project(FrameAlignerMex)

if(DEFINED ENV{MATLAB_VER})
	if(UNIX)
		set(Matlab_ROOT_DIR "/Applications/Matlab_$ENV{MATLAB_VER}.app")
	else()
		set(Matlab_ROOT_DIR "C:/Program Files/MATLAB/$ENV{MATLAB_VER}")
	endif()
endif()
message("Set Matlab root: ${Matlab_ROOT_DIR}")


set(MATLAB_FIND_DEBUG TRUE)
find_package(Matlab COMPONENTS MEX_COMPILER MX_LIBRARY REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_DEBUG_POSTFIX "")

set(SOURCE
   FrameAlignerMex.cpp
)

set(HEADERS
   MexUtils.h
)

matlab_add_mex(NAME FrameAlignerMex SRC ${SOURCE} ${HEADERS} $<TARGET_OBJECTS:FrameAligner>)

include_directories(${FrameAligner_INCLUDE_DIRS})
target_link_libraries(FrameAlignerMex ${FrameAligner_LIBRARIES})

if(FrameAlignerMEX_OUT_DIR)
   add_custom_command(TARGET FrameAlignerMex POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FrameAlignerMex> ${FrameAlignerMEX_OUT_DIR})
endif()
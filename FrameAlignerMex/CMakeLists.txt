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

set(CMAKE_DEBUG_POSTFIX "")

set(SOURCE
   FrameAlignerMex.cpp
)

set(HEADERS
   MexUtils.h
)

matlab_add_mex(NAME FrameAlignerMex SRC ${SOURCE} ${HEADERS} LINK_TO FrameAligner)

if(FrameAlignerMEX_OUT_DIR)
   add_custom_command(TARGET FrameAlignerMex POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FrameAlignerMex> ${FrameAlignerMEX_OUT_DIR})
endif()
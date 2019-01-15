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
find_package(Matlab COMPONENTS MEX_COMPILER MX_LIBRARY)

set(CMAKE_DEBUG_POSTFIX "")

set(SOURCE
   FrameAlignerMex.cpp
)

set(HEADERS
   MexUtils.h
)

if (MATLAB_FOUND)
	matlab_add_mex(NAME FrameAlignerMex SRC ${SOURCE} ${HEADERS} LINK_TO FrameAligner)

	if (FrameAlignerMex_OUT_DIR)
		if (WIN32)
			# For windows, use deployqt to marshall all dlls etc
			add_custom_command(TARGET FrameAlignerMex POST_BUILD COMMAND ${CMAKE_COMMAND} -E
				copy_directory $<TARGET_FILE_DIR:FrameAlignerMex> ${FrameAlignerMex_OUT_DIR})

			if (EXISTS "$ENV{MKLROOT}")
				# install MKL redist's if we used it
				get_filename_component(INTEL_REDIST "$ENV{MKLROOT}/../redist/intel64_win/compiler/" REALPATH)
				add_custom_command(TARGET FrameAlignerMex POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${INTEL_REDIST} ${FrameAlignerMex_OUT_DIR})
			endif()
		else()
			# On mac, just copy the build file
			add_custom_command(TARGET FrameAlignerMex POST_BUILD COMMAND ${CMAKE_COMMAND} -E
				copy $<TARGET_FILE:FrameAlignerMex> ${FrameAlignerMex_OUT_DIR})
		endif()
	endif()
else()
	message(WARNING "Matlab not found; cannot build FrameAlignerMex file")
endif()

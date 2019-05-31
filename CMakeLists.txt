project(CSCSS)
cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost 1.65 COMPONENTS program_options REQUIRED)

include_directories(${Boost_INCLUDE_DIR})

#set(CMAKE_CXX_STANDARD 17)
add_definitions("-g -std=c++1z -Os -mfpmath=sse")
#add_definitions("-g -std=c++11 -Os -march=corei7 -mfpmath=sse")

if(${DEBUG})
    add_definitions("-DDEBUG=1")
    message("DEBUG=1")
else(${DEBUG})
    add_definitions("-DDEBUG=0")
    message("DEBUG=0")
endif(${DEBUG})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_subdirectory(libs/)
add_subdirectory(imgSemanticSeg/)
add_subdirectory(decisionBoundaries/)


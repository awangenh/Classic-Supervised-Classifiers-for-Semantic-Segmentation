include_directories (${CSCSS_SOURCE_DIR}/libs/)
link_directories (${CSCSS_BINARY_DIR}/libs/)

add_executable (decisionBoundaries decisionBoundaries.cpp)

# Link the executable to the library.
target_link_libraries (decisionBoundaries Util_LIBS)
target_link_libraries (decisionBoundaries ${OpenCV_LIBS})
target_link_libraries (decisionBoundaries ${Boost_LIBRARIES})
target_link_libraries (decisionBoundaries Mapping_LIBS)
target_link_libraries (decisionBoundaries ImageFeatures_LIBS)
target_link_libraries (decisionBoundaries stdc++fs)


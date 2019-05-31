include_directories (${CSCSS_SOURCE_DIR}/libs/)
link_directories (${CSCSS_BINARY_DIR}/libs/)

add_executable (imgSeg imgSeg.cpp)

# Link the executable to the library.
target_link_libraries (imgSeg Util_LIBS)
target_link_libraries (imgSeg ${OpenCV_LIBS})
target_link_libraries (imgSeg ${Boost_LIBRARIES})
target_link_libraries (imgSeg Mapping_LIBS)
target_link_libraries (imgSeg ImageFeatures_LIBS)
target_link_libraries (imgSeg stdc++fs)

add_executable (precisionCal precisionCal.cpp)
target_link_libraries (precisionCal Util_LIBS)
target_link_libraries (precisionCal ${OpenCV_LIBS})
target_link_libraries (precisionCal ${Boost_LIBRARIES})
target_link_libraries (precisionCal PrecisionMeasures_LIBS)
target_link_libraries (precisionCal stdc++fs)

set(sourcesList)
list(APPEND sourcesList imgSegPrecisionMeasure.cpp)
list(APPEND sourcesList imgF1Score.cpp)
list(APPEND sourcesList imgIOU.cpp)
add_library(PrecisionMeasures_LIBS ${sourcesList})
target_link_libraries(PrecisionMeasures_LIBS ${OpenCV_LIBS})


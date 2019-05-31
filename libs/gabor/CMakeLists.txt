#building just a library. 
add_library(Gabor_LIBS gabor_texture.cpp)
target_link_libraries(Gabor_LIBS ${OpenCV_LIBS})
target_link_libraries (Gabor_LIBS Util_LIBS)


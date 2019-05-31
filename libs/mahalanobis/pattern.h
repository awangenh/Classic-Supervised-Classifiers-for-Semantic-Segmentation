#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

enum colorSpaceModel { RGB, HSV, HSI, LAB};
//-------------------------------------------------------------------------------------
class point2D {
public:
    int x, y;
    point2D() {};
    point2D(int _x, int _y) : x(_x), y(_y) {};
};
//-------------------------------------------------------------------------------------
class colorRGB {
public:
    unsigned char r, g, b;
    colorRGB() {};
    colorRGB(unsigned char _r, unsigned char _g, unsigned char _b) : r(_r), g(_g), b(_b) {};
};
//-------------------------------------------------------------------------------------
class doubleVector {
public:
    double *v;
    doubleVector() : v(NULL) {};
	  ~doubleVector(){
		free(v);
	}
};
//--------------------------------------------------------------------------------
class pattern {
  
private:
    colorSpaceModel m_colorModel;
    
    //list of point2D's
    point2D *m_coordinates;
    //list of colors
    //colorRGB *m_colors;

	//list of colors
    doubleVector *m_ddata;
	//number of dimensions
    unsigned int m_dim;
	//size of m_list
    unsigned int m_sizeList;
    
public:
    //contructor and destructor
    pattern(int size, int dim);
    pattern(const char *filename="conf.maha");
    ~pattern();
    
    //load a pattern file and store in this class private members
    bool loadfile(const char* filename); 

    //get's for each member variable
    unsigned int getSize() { return m_sizeList; }
    unsigned int getDim() { return m_dim; }
    doubleVector *getData() { return m_ddata; }
    
    //these thwo functions need be proceed  pattern(int size, int dim);
    void setData(doubleVector *data);
    void setCoordinates(point2D *coord);
    
    //set a different color space model
    //void setColorModel(colorSpaceModel colorModel);
};
//--------------------------------------------------------------------------------

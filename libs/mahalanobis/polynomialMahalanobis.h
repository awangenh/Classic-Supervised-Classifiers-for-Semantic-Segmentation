/* 
 * File:   polyMahalanobis.h
 * Author: asobieranski
 *
 * Created on January 21, 2009, 3:27 PM
 * Refactored on August 26, 2009, 23:10 PM
 *    --> converting the functions evaluate and evaluateP2P to c++ insted of matlab
 *
 * Refactored on July 15, 2010, 17:00 PM
 *    --> converting the makeSpace to c++ insted of matlab
 * Refactored on August 29, 2012, 14:00 PM
 *    --> found error on buildind the topological map. Added documentation.

 */

#pragma once

#include <iostream>
#include <malloc.h>
#include <string.h>
#include <opencv/cv.h>
#include "pattern.h"

#include "debug.h"

namespace classifiers
{
    /*! \brief Class that represents the polynomial basis variables
        \param cvm_A_basis covariance matrix of all dimensions
        \param max_aP maximum value of covariance matrix use to normalize the topological map
        \param ind_usesize size of ind_use vector
        \param ind_use vector of non-repeated dimensions
        \param d_proj dimension of the projection
        \param dmssize size of dms vector 
        \param dms for all k in K from S from SVD => -w[k]/(sigma²*(w[k]+sigma²))
        \param sigma_inv small sigma value to avoid inversion limitations
        \param next pointer for the next polynomial dimension
 */
class lev_basis
{
public:
	//these two are obsolete !!!!
	//int H, W;
	//float **A_basis;

	CvMat *cvm_A_basis;
	float max_aP;
	int ind_usesize;
	unsigned int *ind_use;

	int d_proj;
	int dmssize;
	float *dms;
	
	float sigma_inv;
	lev_basis *next;
	
	lev_basis() : cvm_A_basis(NULL), ind_use(NULL), dms(NULL), next(NULL) {};
};
//-----------------------------------------------------------------------------
 /*! \brief Class that represents the polynomial model
        \param m_center center of space
        \param num_levels size of polynomial terms projections
        \param max_level the maximum level achieved by the projections
        \param num_initialdim initial dimension based on pattern
        \param levBegin pointer for the polynomial bases variables
 */
class polyModel
{
public:
	
	float *m_center;
	//size of polynomial terms projections
	int num_levels;
	
	//the maximum level achieved by the projections
	int max_level;
	
	//initial dimension based on pattern
	unsigned int num_initialdim;
	
	//lev basis
	lev_basis *levBegin;
  
public:
	//constructor
	polyModel() : m_center(NULL), levBegin(NULL) {};

	//destructor
	~polyModel() {
		int i=0;
		lev_basis *p=levBegin;
		while(p)
		{
			lev_basis *pfree = p;
			p=p->next;
			
			//dealocating vectors
			cvReleaseMat(&pfree->cvm_A_basis);
			
			free(pfree->ind_use);
			free(pfree->dms);
			free(pfree);
			
			printf("\nLevel %d disalocated !", ++i);
		}
		if(m_center) 
			free(m_center);
		
		printf("\n");
	};
};
//-----------------------------------------------------------------------------
/*! \brief Class that represents the polynomial
        \param m_model an instance of a polynomial model
        \param m_pattern an instance of a pattern
 */
class polyMahalanobis
{
private:
	//an instance of a polynomial model
	polyModel *m_model;
	
	//an instance of a pattern 
	pattern *m_pattern;
	
	/*! \brief factorial calculation
            \param n value to be calculate the factorial
            \result factorial value
         */
        float fat(int n) { if(n==0) return 1; else return n*fat(n-1); }
    
        //calc a mean of a d-dimensional vector at each component
	float *calc_mean(doubleVector *data, unsigned int size, unsigned int d);
	
	//get the maximum value from a given vector
	float getMaxValue(float *in, unsigned int size);
	
	//get the maximum value from a given vector
	float getMaxAbsValue(float *in, unsigned int size);

	//similar to find in matlab
	//find in a vetor values equal to zero (0), lower (-1) or bigger than (1), and return an allocated array of indexes
	unsigned int *find_eq(int opt, float *in, unsigned int size, unsigned int &lenght);
	
	//similar to nchoosek in matlab 
    unsigned int *nchoosek(unsigned int proj, unsigned int &size);
	
	//compute next projection
	CvMat *newProjection(CvMat *A, CvMat *cross);

	//compute the variance of a CvMat column, if c==-1, all columns are a single vector
	float calcVariance(CvMat *A, int column);
	//compute the variance of each CvMat column, and return in var_dim
	void calcVariance(CvMat *A, float *var_dim);
	//remove null dimensions, e.g.lower than a variance error, and return a new one
	CvMat *removeNullDimensions(CvMat *A, unsigned int *ind_use);
	//fits the matrix A only to the ind_use vector
	CvMat *removeNullIndexes(CvMat *A, unsigned int *ind_use, unsigned int size);
	
	// *** DEBUG FLAGS ***
	void showLevBasis();
	void showMatrixValues(CvMat *A);
	void showInputData(double *im_data, unsigned int size, unsigned int dimensions);
	void showResultingData(double *res_data, unsigned int size, unsigned int levels);
  
public:
	//constructor and destructor
	polyMahalanobis();  
	~polyMahalanobis();
	
	//*** training step ***
	//seting a pattern structure
	bool setPattern(pattern *_pattern);
	//making a space and storing in 
	bool makeSpace(unsigned int order);

	//*** testing points or list of points ***
	//compare an im_data array of double's with size N to the center of the space
	//for the both cases it returns a double vector of dimension q-order.

	//evaluate functions
    void evaluateToCenter(double *im_data, unsigned int size, unsigned int dimensions, double* output_m_intensValues);
    void evaluateToVector(double *im_data, double *refVector, unsigned int size, unsigned int dimensions, double* output_m_intensValues);
	 
	
	/*! \brief get the maximum polynomial order of a created space
            \result the mean of the selected points
         */
        int getMaxqOrder() {return m_model->num_levels;}

	/*! \brief get the center of the space
            \result center of the space
         */
	float *getCenter() { return m_model->m_center; }

	
        /*! \brief check if the number of levels accomplished
            \result center of the space
         */
        inline bool isSampled() { return (m_model->num_levels > -1) ? 1 : 0; };
	
	
        /*! \brief check if there is a istancied pattern
            \result true if exists an instancied pattern
         */
        bool hasPattern() { return (m_pattern) ? true : false; }
};       
//-----------------------------------------------------------------------------
};



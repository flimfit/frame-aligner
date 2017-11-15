#pragma once

#include <complex>
#include <fftw3.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Pool.h"

class CorrelatorPlan
{
public:
   
   CorrelatorPlan(const std::vector<int>& dim);
   ~CorrelatorPlan();

   fftw_plan plan, inv_plan;
   std::vector<double> in;
   std::vector<std::complex<double> > out;
};


class VolumePhaseCorrelator
{
public:
 
   VolumePhaseCorrelator(int n_z, int n_y, int n_x);
   
   void setReference(const float* volume);
   cv::Point3d computeShift(const float* volume);

private:

   void computeWindow();
   std::vector<double> hann(int n);

   std::vector<int> dims;
   int n_el, n_complex;
   
   std::vector<double> window;   
   std::vector<std::complex<double> > reference;

   Pool<CorrelatorPlan, std::vector<int>> pool;
};
#pragma once

#include <complex>
#include <fftw3.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stack>
#include <mutex>

class CorrelatorPlan
{
public:
   
   CorrelatorPlan(const std::vector<int>& dim);
   ~CorrelatorPlan();

   fftw_plan plan, inv_plan;
   std::vector<double> in;
   std::vector<std::complex<double> > out;
};

template<class T, class U>
class Pool
{
public:

   Pool(const U& init) : init(init)
   { 
      pool.push( std::make_unique<T>(init) );
   }

   std::unique_ptr<T> get()
   {
      std::lock_guard<std::mutex> lk(m);
      
      if (pool.empty())
         return std::make_unique<T>(init);
      
      auto tmp = std::move(pool.top());
      pool.pop();
      
      return tmp;
   }

   void release(std::unique_ptr<T> el)
   {
      std::lock_guard<std::mutex> lk(m);
      pool.push(std::move(el));
   }

private:
   std::stack<std::unique_ptr<T>> pool;
   std::mutex m;
   U init;
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
#pragma once
#include <opencv2/opencv.hpp>
#include <dlib/optimization.h>
#include "Cv3dUtils.h"
#include <vector>
#include "ImageScanParameters.h"

template <typename T>
bool isValidPoint(const cv::Point3_<T>& pt, const std::vector<int>& dims)
{
   bool validXY = 
         (pt.x >= 0) &&
         (pt.y >= 0) && 
         (pt.x <= (dims[X] - 1)) &&
         (pt.y <= (dims[Y] - 1));

   if (dims[Z] > 1)
      return validXY && (pt.z >= 0) && (pt.z <= (dims[Z] - 1));
   else 
      return validXY;            
}

template <typename T>
bool isValidPoint(const cv::Vec<T,3>& pt, const std::vector<int>& dims)
{
   bool validXY = 
         (pt[X] >= 0) &&
         (pt[Y] >= 0) && 
         (pt[X] <= (dims[X] - 1)) &&
         (pt[Y] <= (dims[Y] - 1));

   if (dims[Z] > 1)
      return validXY && (pt[Z] >= 0) && (pt[Z] <= (dims[Z] - 1));
   else 
      return validXY;            
}


template <typename T>
T getZeroPadded(const cv::Mat& img, int z, int y, int x)
{
   bool valid =
      (x >= 0) && (x < img.size[X]) &&
      (y >= 0) && (y < img.size[Y]) &&
      ((img.size[Z] == 1) || ((z >= 0) && (z < img.size[Z])));
   return (valid) ? img.at<T>(z, y, x) : 0.0f;
}

class Range
{
public:
   int begin;
   int end;

   int interval() { return end - begin; }
};

template<class T>
class OffsetVector
{
public: 

   OffsetVector<T>(size_t i0_ = 0, size_t i1_ = 1)
   {
      i0 = i0_;
      i1 = i1_;
      n = i1 - i0;
      if (i0 > i1) n = 0;
      v.resize(n);
   }

   T at(int i) const
   {
      if ((i < i0) || (i > i1)) 
         return T();
      return v[i - i0];
   }

   T& operator[](int i) { return v[i - i0]; }

   size_t first() { return i0; }
   size_t last() { return i1; }
   size_t size() { return n; }

private:
   std::vector<T> v;
   size_t i0;
   size_t i1;
   size_t n;
};

using namespace dlib;
typedef matrix<double, 0, 1> column_vector;

class AbstractFrameWarper
{
public:

   virtual ~AbstractFrameWarper() {};

   void setReference(const cv::Mat& reference, int nD, const ImageScanParameters& image_params);

   virtual double getError(const cv::Mat& img, const std::vector<cv::Point3d>& D) = 0;
   virtual void getJacobian(const cv::Mat& img, const std::vector<cv::Point3d>& D, column_vector& jac) = 0;

   virtual void warpImageInterpolated(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value = 0) = 0;
   virtual void warpImage(const cv::Mat& img, cv::Mat& wimg, cv::Mat& coverage, const std::vector<cv::Point3d>& D);
   virtual void warpImageNormalised(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int cv_type = CV_8U);

   cv::Point3d warpPoint(const std::vector<cv::Point3d>& D, int x, int y, int z, int spatial_binning = 1);
   
   int n_dim = 2;
   std::vector<int> dims;

   virtual void registerFrame(const cv::Mat& frame) {};
   virtual void deregisterFrame(const cv::Mat& frame) {};

   virtual void clearTemp() {};

protected:

   ImageScanParameters processScanParameters(const ImageScanParameters& image_params);

   void precomputeInterp(const ImageScanParameters& image_params);
   void computeSteepestDecentImages(const cv::Mat& frame);
   double computeHessianEntry(int pi, int pj);
   void computeHessian();
   void computeJacobian(const cv::Mat& error_img, column_vector& jac);
   
   virtual void setupReferenceInformation() {};

   cv::Mat Di;
   cv::Mat Df;
   matrix<double> H;
   std::vector<OffsetVector<cv::Point3f>> VI_dW_dp;
   std::vector<Range> D_range;
   
   int nD;
   cv::Mat reference;
   ImageScanParameters image_params;

   friend class OptimisationModel;
};

class CpuFrameWarper : public AbstractFrameWarper
{
public:

   double getError(const cv::Mat& img, const std::vector<cv::Point3d>& D);
   void getJacobian(const cv::Mat& img, const std::vector<cv::Point3d>& D, column_vector& jac);

   void warpImageInterpolated(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value = 0);
   
protected:

   double computeErrorImage(cv::Mat& wimg, cv::Mat& error_img);

   std::map<void*, cv::Mat> error_buffer;
   
};
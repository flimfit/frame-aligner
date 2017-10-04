#pragma once

#include "AbstractFrameAligner.h"
#include "VolumePhaseCorrelator.h"
#include <functional>
#include <array>
#include <opencv2/opencv.hpp>
#include <dlib/optimization.h>

#define X 2
#define Y 1
#define Z 0


using namespace dlib;
typedef matrix<double, 0, 1> column_vector;

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

   OffsetVector<T>(size_t n_ = 0, size_t i0_ = 0)
   {
      i0 = i0_;
      n = n_;
      v.resize(n);
   }

   T at(int i) 
   {
      if ((i < i0) || (i >= (i0 + n))) return 0;
      return v[i - i0];
   }

   T& operator[](int i) { return v[i - i0]; }

private:
   std::vector<T> v;
   size_t i0;
   size_t n;
};

class FrameWarpAligner : public AbstractFrameAligner
{
public:

   FrameWarpAligner(RealignmentParameters params);
   
   ~FrameWarpAligner() {}

   bool empty() { return false; };
   void clear() { Dstore.clear(); };

   RealignmentType getType() { return RealignmentType::Warp; };

   void setNumberOfFrames(int n_frame);
   void setReference(int frame_t, const cv::Mat& reference_);
   RealignmentResult addFrame(int frame_t, const cv::Mat& frame);
   void shiftPixel(int frame_t, double& x, double& y, double& z);
   double getFrameCorrelation(int frame_t) { return results[frame_t].correlation; };
   double getFrameCoverage(int frame_t) { return results[frame_t].coverage; };
   void reprocess();

   bool frameReady(int frame);

   void writeRealignmentInfo(std::string filename);

protected:

   void precomputeInterp();
   void computeSteepestDecentImages(const cv::Mat& frame);
   double computeHessianEntry(int pi, int pj);
   void computeHessian();
   void computeJacobian(const cv::Mat& error_img, column_vector& jac);
   void warpImage(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value = 0);
   void warpImageIntensityPreserving(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D);
   void warpCoverage(cv::Mat& coverage, const std::vector<cv::Point3d>& D);
   cv::Point3d warpPoint(const std::vector<cv::Point3d>& D, int x, int y, int z, int spatial_binning = 1);
   double computeErrorImage(cv::Mat& wimg, cv::Mat& error_img);
   
   void smoothStack(const cv::Mat& in, cv::Mat& out);
   cv::Mat reshapeForOutput(cv::Mat& m);
   cv::Mat reshapeForProcessing(cv::Mat& m);

   template <typename T>
   bool isValidPoint(const cv::Point3_<T>& pt)
   {
      int validXY = 
            (pt.x >= 0) &&
            (pt.y >= 0) && 
            (pt.x <= (dims[X] - 1)) &&
            (pt.y <= (dims[Y] - 1));

      if (n_dim == 3)
         return validXY && (pt.z >= 0) && (pt.z <= (dims[Z] - 1));
      else 
         return validXY;            
   }

   template <typename T>
   bool isValidPoint(const cv::Vec<T,3>& pt)
   {
      int validXY = 
            (pt[X] >= 0) &&
            (pt[Y] >= 0) && 
            (pt[X] <= (dims[X] - 1)) &&
            (pt[Y] <= (dims[Y] - 1));

      if (n_dim == 3)
         return validXY && (pt[Z] >= 0) && (pt[Z] <= (dims[Z] - 1));
      else 
         return validXY;            
   }
   
   std::vector<Range> D_range;

   cv::Mat smoothed_reference;
   cv::Mat sum_1, sum_2;

   cv::Mat Di;
   cv::Mat Df;
   matrix<double> H;

   cv::Point3d Dlast;

   std::vector<std::vector<cv::Point3d>> Dstore;
   std::vector<RealignmentResult> results;

   int n_dim = 2;
   int nD = 10;

   int phase_downsampling = 4;

   int n_x_binned;
   int n_y_binned;

   std::vector<int> dims;
   bool output2d = false;

   std::vector<OffsetVector<float>> VI_dW_dp_x, VI_dW_dp_y, VI_dW_dp_z;

   std::unique_ptr<VolumePhaseCorrelator> phase_correlator;

   friend class OptimisationModel;
};


class OptimisationModel
{
   /*!
   This object is a "function model" which can be used with the
   find_min_trust_region() routine.
   !*/

public:
   typedef ::column_vector column_vector;
   typedef matrix<double> general_matrix;

   OptimisationModel(FrameWarpAligner* aligner, const cv::Mat& frame, const cv::Mat& raw_frame);

   double operator() (const column_vector& x) const;
   void get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const;

   cv::Mat getMask(const column_vector& x);
   cv::Mat getWarpedRawImage(const column_vector& x);
   cv::Mat getWarpedImage(const column_vector& x);

protected:

   FrameWarpAligner* aligner;
   cv::Mat raw_frame;
   cv::Mat frame;
   RealignmentParameters realign_params;
};

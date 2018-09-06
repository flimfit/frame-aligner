#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include "ImageScanParameters.h"
#include "CvCache.h"


template<>
size_t Cache<cv::Mat>::getSize(const cv::Mat& obj);

typedef std::shared_ptr<CachedObject<cv::Mat>> CachedMat;

enum class DefaultReferenceFrame
{
   FirstFrame = 0,
   MiddleFrame = 1,
   LastFrame = 2
};

enum class RealignmentType
{
   None        = 0,
   Translation = 1,
   Warp        = 2
};

std::string realignmentTypeString(RealignmentType t);

cv::Mat downsample(const cv::Mat& im1, int factor);

class RealignmentParameters
{
public:
   RealignmentType type = RealignmentType::None;
   int spatial_binning = 1;
   int frame_binning = 1;
   int n_resampling_points = 30;
   bool reprocessing = false;
   double smoothing = 0;
   double correlation_threshold = 0;
   double coverage_threshold = 0;
   bool prefer_gpu = true;
   bool store_frames = true;
   DefaultReferenceFrame default_reference_frame = DefaultReferenceFrame::FirstFrame;

   bool use_realignment() { return type != RealignmentType::None; }
   bool use_rotation() { return false; } //{ return type == RealignmentType::RigidBody; }
};

class RealignmentResult
{
public:
   CachedMat frame;
   CachedMat realigned;
   CachedMat realigned_preserving;
   CachedMat mask;
   double correlation = 1;
   double unaligned_correlation = 1;
   double coverage = 1;
   bool done = false;

   bool useFrame(const RealignmentParameters& p) const
   {
      return ((correlation >= p.correlation_threshold) && (coverage >= p.coverage_threshold));
   }
};

class AbstractFrameAligner
{
public:

   virtual ~AbstractFrameAligner() {};

   static AbstractFrameAligner* createFrameAligner(RealignmentParameters params);

   virtual void clear() = 0;

   bool frameValid(int frame) { return true; } // worse case will just use last
   virtual bool frameReady(int frame) = 0;

   virtual RealignmentType getType() = 0;

   void setRealignmentParams(RealignmentParameters params_) { realign_params = params_; }
   void setImageScanParams(ImageScanParameters params_) { image_params = params_; }
   virtual void setReference(int frame_t, const cv::Mat& reference_) = 0;
   virtual void addFrame(int frame_t, CachedMat frame) = 0; // should return aligned frame
   virtual cv::Mat realignAsFrame(int frame_t, const cv::Mat& frame, bool interpolate_missing = true) = 0; // should realign provided frame as if it was frame_t
   virtual void shiftPixel(int frame_t, double& x, double& y, double& z) = 0;
   virtual double getFrameCorrelation(int frame_t) { return 1.; }
   virtual double getFrameCoverage(int frame_t) { return 1.; }
   virtual void setNumberOfFrames(int n_frames_) { n_frames = n_frames_; }

   virtual void writeRealignmentInfo(std::string filename) {};

   virtual void reprocess() {};

   virtual void clearTemp() {};

   const std::map<size_t, RealignmentResult>& getRealignmentResults() { return results; }
   
   RealignmentResult getRealignmentResult(size_t frame) 
   {
      auto it = results.find(frame);
      if (it != results.end())
         return results.find(frame)->second;
      else
         return RealignmentResult();
   }

protected:
   RealignmentParameters realign_params;
   ImageScanParameters image_params;
   cv::Mat reference;
   int n_frames = 1;

   std::map<size_t,RealignmentResult> results;
};

class NullFrameAligner : public AbstractFrameAligner
{
public:

   //bool empty() {  };
   void clear() {};
   bool frameReady(int frame) { return true; };

   RealignmentType getType() { return RealignmentType::None; };
   void setReference(int frame_t, const cv::Mat& reference_) {};
   void addFrame(int frame_t, CachedMat frame) {}; // should return aligned frame
   cv::Mat realignAsFrame(int frame_t, const cv::Mat& frame, bool interpolate_missing = true) { return frame; }; // should realign provided frame as if it was frame_t
   void shiftPixel(int frame_t, double& x, double& y, double& z) {};
};
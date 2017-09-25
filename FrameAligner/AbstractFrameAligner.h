#pragma once
#include <opencv2/opencv.hpp>
#include <memory>

enum class RealignmentType
{
   None        = 0,
   Translation = 1,
   RigidBody   = 2,
   Warp        = 3
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

   bool use_realignment() { return type != RealignmentType::None; }
   bool use_rotation() { return type == RealignmentType::RigidBody; }
};

class RealignmentResult
{
public:
   cv::Mat frame;
   cv::Mat realigned;
   cv::Mat realigned_preserving;
   cv::Mat mask;
   double correlation = 0;
   double unaligned_correlation = 0;
   double coverage = 0;
   bool done = false;
};

class ImageScanParameters
{
public:
   ImageScanParameters(double line_duration = 100, double interline_duration = 101, double interframe_duration = 102, int n_x = 1, int n_y = 1, int n_z = 1, bool bidirectional = false) :
      line_duration(line_duration), interline_duration(interline_duration), interframe_duration(interframe_duration), n_x(n_x), n_y(n_y), n_z(n_z)
   {
      n_x = std::max(n_x, 1);
      n_y = std::max(n_y, 1);

      if (interline_duration <= line_duration)
         interline_duration = 1.01 * line_duration;

      pixel_duration = line_duration / n_x;
      frame_duration = n_y * interline_duration;
   }

   double line_duration;
   double interline_duration;
   double pixel_duration;
   double frame_duration;
   double interframe_duration;
   int n_x;
   int n_y;
   int n_z;
   bool bidirectional;
};

class AbstractFrameAligner
{
public:

   virtual ~AbstractFrameAligner() {};

   static AbstractFrameAligner* createFrameAligner(RealignmentParameters params);

   virtual bool empty() = 0;
   virtual void clear() = 0;

   bool frameValid(int frame) { return true; } // worse case will just use last
   virtual bool frameReady(int frame) = 0;

   virtual RealignmentType getType() = 0;

   void setRealignmentParams(RealignmentParameters params_) { realign_params = params_; }
   void setImageScanParams(ImageScanParameters params_) { image_params = params_; }
   virtual void setReference(int frame_t, const cv::Mat& reference_) = 0;
   virtual RealignmentResult addFrame(int frame_t, const cv::Mat& frame) = 0; // shold return aligned frame
   virtual void shiftPixel(int frame_t, double& x, double& y, double& z) = 0;
   virtual double getFrameCorrelation(int frame_t) { return 0.; }
   virtual double getFrameCoverage(int frame_t) { return 0.; }
   virtual void setNumberOfFrames(int n_frames_) { n_frames = n_frames_; }

   virtual void writeRealignmentInfo(std::string filename) {};

   virtual void reprocess() {};

   RealignmentResult& getResult(int frame)
   {
      if (frame >= n_frames) throw std::runtime_error("Frame index too large");
      return results[frame];
   }

protected:
   RealignmentParameters realign_params;
   ImageScanParameters image_params;
   cv::Mat reference;
   int n_frames = 1;

   std::vector<RealignmentResult> results;
};


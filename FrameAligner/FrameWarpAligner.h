#pragma once

#include "AbstractFrameAligner.h"
#include "VolumePhaseCorrelator.h"
#include <functional>
#include <array>
#include <memory>
#include <opencv2/opencv.hpp>
#include <dlib/optimization.h>
#include "Cv3dUtils.h"

#include "FrameWarper.h"
#include "GpuFrameWarper.h"

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
   cv::Mat realignAsFrame(int frame_t, const cv::Mat& frame);
   void shiftPixel(int frame_t, double& x, double& y, double& z);
   double getFrameCorrelation(int frame_t) { return results[frame_t].correlation; };
   double getFrameCoverage(int frame_t) { return results[frame_t].coverage; };
   void reprocess();

   bool frameReady(int frame);

   void writeRealignmentInfo(std::string filename);

protected:

   void smoothStack(const cv::Mat& in, cv::Mat& out);
   cv::Mat reshapeForOutput(cv::Mat& m);
   cv::Mat reshapeForProcessing(cv::Mat& m);
   
   cv::Mat smoothed_reference;
   cv::Mat sum_1, sum_2;
   
   std::shared_ptr<AbstractFrameWarper> warper;
   std::shared_ptr<AbstractFrameWarper> alt_warper;


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


   std::unique_ptr<VolumePhaseCorrelator> phase_correlator;

   std::list<std::pair<std::vector<cv::Point3d>, cv::Mat>> error_buffer;

   std::mutex align_mutex;
   std::condition_variable align_cv;

   friend class OptimisationModel;
};

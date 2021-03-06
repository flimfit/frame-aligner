#pragma once

#include <fftw3.h>

#include "AbstractFrameAligner.h"
#include <atomic>
#include <map>

class Transform
{
public:

   Transform(double frame_ = 0)
   {
      frame = frame_;
      angle = 0.0;
      shift.x = 0.0;
      shift.y = 0.0;
   }

   Transform(double frame, double angle, cv::Point2d shift) :
      frame(frame), angle(angle), shift(shift)
   {
   }

   double frame;
   double angle;
   cv::Point2d shift;
};

class RigidFrameAligner : public AbstractFrameAligner
{
public:

   RigidFrameAligner(RealignmentParameters params);

   bool empty();
   void clear();

   bool frameReady(int frame)
   {
      return frame_transform.find(frame + 1) != frame_transform.end();
   }

   RealignmentType getType() { return realign_params.type; };

   void setReference(int frame_t, const cv::Mat& reference_);
   void addFrame(int frame_t, CachedMat frame);
   cv::Mat realignAsFrame(int frame_t, const cv::Mat& frame) { return frame; } // TODO
   void shiftPixel(int frame, double& x, double& y, double& z);
   
   cv::Point2d getRigidShift(int frame);

private:

   void addTransform(int frame_t, Transform t);
   void interpolate(Transform& t1, Transform& t2, double frame, cv::Mat& affine, cv::Point2d& shift);
   void getAffine(double frame, cv::Mat& affine, cv::Point2d& shift);

   std::map<size_t,Transform> frame_transform;

   double cache_frame = -1;
   cv::Mat cache_affine;
   cv::Point2d cache_shift;

   cv::Point2d centre;
   cv::Point2d centre_binned;

   cv::Mat reference;
   cv::Mat window;

   cv::Mat log_polar0;

   std::atomic<int> frames_complete;
};
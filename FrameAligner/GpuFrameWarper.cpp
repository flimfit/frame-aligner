#include "GpuFrameWarper.h"
#include "GpuFrameWarperKernels.h"

// Includes CUDA
#include <cuda_runtime.h>

#include "helper_cuda.h"



void GpuFrameWarper::registerFrame(const cv::Mat& frame)
{
   frames.push_back(GpuFrame(frame));
}

void GpuFrameWarper::deregisterFrame(const cv::Mat& frame)
{
   std::remove_if(frames.begin(), frames.end(), [&](auto& it) { return it.isSame(frame); } );
}


void GpuFrameWarper::setupReferenceInformation()
{

}

double GpuFrameWarper::computeErrorImage(cv::Mat& wimg, cv::Mat& error_img)
{
   return 0;
}

void GpuFrameWarper::computeJacobian(const cv::Mat& error_img, column_vector& jac)
{
   
}

void GpuFrameWarper::warpImage(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value)
{}

void GpuFrameWarper::warpImageIntensityPreserving(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D)
{}

void GpuFrameWarper::warpCoverage(cv::Mat& coverage, const std::vector<cv::Point3d>& D)
{}

cv::Point3d GpuFrameWarper::warpPoint(const std::vector<cv::Point3d>& D, int x, int y, int z, int spatial_binning)
{
   return cv::Point3d(0,0,0);
}
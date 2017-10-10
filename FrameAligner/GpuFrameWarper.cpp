#include "GpuFrameWarper.h"
#include "GpuFrameWarperKernels.h"

// Includes CUDA
#include <cuda_runtime.h>

#include "helper_cuda.h"



void GpuFrameWarper::registerFrame(const cv::Mat& frame)
{
   frames[frame.data] = std::make_shared<GpuFrame>(frame, nD);
}

void GpuFrameWarper::deregisterFrame(const cv::Mat& frame)
{
   frames.erase(frame.data);
}


void GpuFrameWarper::setupReferenceInformation()
{

   float3 offset;
   double stack_duration = dims[Z] * image_params.frame_duration;
   offset.x = image_params.pixel_duration / stack_duration;
   offset.y = image_params.interline_duration / stack_duration;
   offset.z = image_params.frame_duration / stack_duration;


   int range_max = std::max_element(VI_dW_dp.begin(), VI_dW_dp.end(), [](auto& a, auto& b) { return a.size() < b.size(); })->size();
   gpu_reference = std::make_unique<GpuReferenceInformation>(reference, offset, nD, range_max);

   float3 *VI_dW_dp_host;
   checkCudaErrors(cudaMallocHost(&VI_dW_dp_host, range_max * nD));

   for (int i = 0; i < nD; i++)
   {
      auto& dp = VI_dW_dp[i];
      int p0 = dp.first();
      for (int p = 0; p < dp.size(); p++)
      {
         VI_dW_dp_host[i*range_max + p].x = dp[p + p0].x;
         VI_dW_dp_host[i*range_max + p].y = dp[p + p0].y;
         VI_dW_dp_host[i*range_max + p].z = dp[p + p0].z;
      }
   }

   GpuRange *range_host = reinterpret_cast<GpuRange*>(D_range.data());

   cudaMemcpy(gpu_reference->VI_dW_dp, VI_dW_dp_host, range_max * nD * sizeof(float3), cudaMemcpyHostToDevice);
   cudaMemcpy(gpu_reference->range, range_host, nD * sizeof(GpuRange), cudaMemcpyHostToDevice);

   checkCudaErrors(cudaFreeHost(VI_dW_dp_host));
}


double GpuFrameWarper::getError(const cv::Mat& frame, const std::vector<cv::Point3d>& D)
{
   return computeError(frames[frame.data].get(), gpu_reference.get());
}

void GpuFrameWarper::getJacobian(const cv::Mat& img, const std::vector<cv::Point3d>& D, column_vector& jac)
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
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
   checkCudaErrors(cudaMallocHost((void**) &VI_dW_dp_host, range_max * nD * sizeof(float3)));

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

   std::vector<GpuRange> range_host(nD);

   for(int i=0; i<nD; i++)
   {
      range_host[i].begin = VI_dW_dp[i].first();
      range_host[i].end = VI_dW_dp[i].last();
   }

   checkCudaErrors(cudaMemcpy(gpu_reference->VI_dW_dp, VI_dW_dp_host, range_max * nD * sizeof(float3), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(gpu_reference->range, range_host.data(), nD * sizeof(GpuRange), cudaMemcpyHostToDevice));

   checkCudaErrors(cudaFreeHost(VI_dW_dp_host));
}

std::shared_ptr<GpuFrame> GpuFrameWarper::getRegisteredFrame(const cv::Mat& frame)
{
   if (frames.count(frame.data) == 0)
   throw std::runtime_error("Unrecognised frame");
   return frames[frame.data];
   }

std::vector<float3> GpuFrameWarper::D2float3(const std::vector<cv::Point3d>& D)
{
   std::vector<float3> Df(nD);
   for(int i=0; i<nD; i++)
   {
      Df[i].x = D[i].x;
      Df[i].y = D[i].y;
      Df[i].z = D[i].z;
   }
   return std::move(Df);
}



double GpuFrameWarper::getError(const cv::Mat& frame, const std::vector<cv::Point3d>& D)
{
   auto f = getRegisteredFrame(frame);
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(f->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   return computeError(f.get(), gpu_reference.get());
}

void GpuFrameWarper::getJacobian(const cv::Mat& frame, const std::vector<cv::Point3d>& D, column_vector& jac)
{
   auto f = getRegisteredFrame(frame);
   //auto Df = D2float3(D); => same as before

   std::vector<float3> jacv = computeJacobian(f.get(), gpu_reference.get());

   jac.set_size(nD * n_dim);
   for(int i=0; i<nD; i++)
   {
      jac(i*n_dim) = jacv[i].x;
      jac(i*n_dim+1) = jacv[i].y;
      if (n_dim==3)
         jac(i*n_dim+2) = jacv[i].z;
   }

}


void GpuFrameWarper::warpImage(const cv::Mat& frame, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value)
{      
   auto f = getRegisteredFrame(frame);
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(f->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   computeWarp(f.get(), gpu_reference.get());

   wimg = cv::Mat(dims, CV_32F);
   checkCudaErrors(cudaMemcpy(wimg.data, f->error_image, dims[X]*dims[Y]*dims[Z]*sizeof(float), cudaMemcpyDeviceToHost));   
}

void GpuFrameWarper::warpImageIntensityPreserving(const cv::Mat& frame, cv::Mat& wimg, cv::Mat& coverage, const std::vector<cv::Point3d>& D)
{
   auto f = getRegisteredFrame(frame);
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(f->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   computeIntensityPreservingWarp(f.get(), gpu_reference.get());

   wimg = cv::Mat(dims, CV_32F);   
   checkCudaErrors(cudaMemcpy(wimg.data, f->error_image, dims[X]*dims[Y]*dims[Z]*sizeof(float), cudaMemcpyDeviceToHost));   

   coverage = cv::Mat(dims, CV_16U);   
   checkCudaErrors(cudaMemcpy(coverage.data, f->mask, dims[X]*dims[Y]*dims[Z]*sizeof(uint16_t), cudaMemcpyDeviceToHost));   
}

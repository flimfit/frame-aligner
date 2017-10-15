#include "GpuFrameWarper.h"
#include "GpuFrameWarperKernels.h"

// Includes CUDA
#include <cuda_runtime.h>

#include "helper_cuda.h"


GpuFrameWarper::GpuFrameWarper()
{
   if (!hasSupportedGpu())
      throw std::runtime_error("No supported GPU");
}

bool GpuFrameWarper::hasSupportedGpu()
{
   return checkCudaCapabilities(3, 0);
}

void GpuFrameWarper::registerFrame(const cv::Mat& frame)
{
   std::lock_guard<std::mutex> lk(mutex);   
   frames[frame.data] = std::make_shared<GpuFrame>(frame);
}
 
void GpuFrameWarper::deregisterFrame(const cv::Mat& frame)
{
   std::lock_guard<std::mutex> lk(mutex);   
   frames.erase(frame.data);
}

void GpuFrameWarper::setupReferenceInformation()
{
   size_t free_mem, total_mem;
   checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
   std::cout << "GPU Memory (total/free) : " << (total_mem / (1024 * 1024)) << " / " << (free_mem / (1024 * 1024)) << " Mb\n";

   size_t required_for_jacobian = 10 * 4 * area(reference);
   stream_VI = (free_mem < required_for_jacobian);

   stream_VI = true;

   if (stream_VI)
      std::cout << "Streaming Jacobian (needed " << required_for_jacobian / (1024 * 1024) << " Mb)\n";
      
   float3 offset;
   double stack_duration = dims[Z] * image_params.frame_duration;
   offset.x = image_params.pixel_duration / stack_duration;
   offset.y = image_params.interline_duration / stack_duration;
   offset.z = image_params.frame_duration / stack_duration;

   range_max = std::max_element(VI_dW_dp.begin(), VI_dW_dp.end(), [](auto& a, auto& b) { return a.size() < b.size(); })->size();

   gpu_reference = std::make_unique<GpuReferenceInformation>(reference, offset, nD, range_max, stream_VI);

   float3* VI_dW_dp_host = gpu_reference->VI_dW_dp_host;
   gpu_reference->range.resize(nD);
   
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
      gpu_reference->range[i].begin = dp.first();
      gpu_reference->range[i].end = dp.last();   

   }   

   GpuWorkingSpaceParams params;
   params.volume = area(reference);
   params.nD = nD;
   params.range_max = range_max;
   pool.setInit(params);

    
   /*
   if (!stream_VI)
   {
      checkCudaErrors(cudaMemcpy(gpu_reference->VI_dW_dp, VI_dW_dp_host, range_max * nD * sizeof(float3), cudaMemcpyHostToDevice));      
      checkCudaErrors(cudaFreeHost(VI_dW_dp_host));
      VI_dW_dp_host = nullptr;
   }
   */

}

std::shared_ptr<GpuFrame> GpuFrameWarper::getRegisteredFrame(const cv::Mat& frame)
{
   std::lock_guard<std::mutex> lk(mutex);   
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
   auto w = pool.get();   
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(w->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   return computeError(f.get(), w.get(), gpu_reference.get());
}

void GpuFrameWarper::getJacobian(const cv::Mat& frame, const std::vector<cv::Point3d>& D, column_vector& jac)
{
   auto f = getRegisteredFrame(frame);
   auto w = pool.get();
   //auto Df = D2float3(D); => same as before

   //if (compute_jacobian_on_gpu)
   {
      std::vector<float3> jacv = computeJacobianGpu(f.get(), w.get(), gpu_reference.get());
      
         jac.set_size(nD * n_dim);
         for(int i=0; i<nD; i++)
         {
            jac(i*n_dim) = jacv[i].x;
            jac(i*n_dim+1) = jacv[i].y;
            if (n_dim==3)
               jac(i*n_dim+2) = jacv[i].z;
         }
   }
   //else
   //{
   //   cv::Mat error_image(dims, CV_32F);
   //   checkCudaErrors(cudaMemcpy(error_image.data, w->error_image, dims[X] * dims[Y] * dims[Z] * sizeof(float), cudaMemcpyDeviceToHost));
   //   computeJacobian(error_image, jac);   
   //}


   /*
   column_vector jac2;
   cv::Mat error_image(dims, CV_32F);
   checkCudaErrors(cudaMemcpy(error_image.data, w->error_image, dims[X] * dims[Y] * dims[Z] * sizeof(float), cudaMemcpyDeviceToHost));
   computeJacobian(error_image, jac2);   

   std::cout << "=============\n";
   std::cout << std::scientific << std::setw(5) << std::setprecision(3) << std::showpos;
   for(int i=0; i<jac.size(); i++)
   {
      if (i % 3 == 0) std::cout << "\n     > " << i /3 << "  |  ";
      std::cout << "(" << jac(i) - jac2(i) << ")  |  ";
   }
   std::cout << "\n";
   */
   
}


void GpuFrameWarper::warpImage(const cv::Mat& frame, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value)
{      
   auto f = getRegisteredFrame(frame);
   auto w = pool.get();
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(w->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   computeWarp(f.get(), w.get(), gpu_reference.get());

   wimg = cv::Mat(dims, CV_32F);
   checkCudaErrors(cudaMemcpy(wimg.data, w->error_image, dims[X]*dims[Y]*dims[Z]*sizeof(float), cudaMemcpyDeviceToHost));   
}

void GpuFrameWarper::warpImageIntensityPreserving(const cv::Mat& frame, cv::Mat& wimg, cv::Mat& coverage, const std::vector<cv::Point3d>& D)
{
   auto f = getRegisteredFrame(frame);
   auto w = pool.get();   
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(w->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   computeIntensityPreservingWarp(f.get(), w.get(), gpu_reference.get());

   wimg = cv::Mat(dims, CV_32F);   
   checkCudaErrors(cudaMemcpy(wimg.data, w->error_image, dims[X]*dims[Y]*dims[Z]*sizeof(float), cudaMemcpyDeviceToHost));   

   coverage = cv::Mat(dims, CV_16U);   
   checkCudaErrors(cudaMemcpy(coverage.data, w->mask, dims[X]*dims[Y]*dims[Z]*sizeof(uint16_t), cudaMemcpyDeviceToHost));   
}

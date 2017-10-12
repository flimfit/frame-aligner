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


int GpuFrameWarper::registerWorkingSpace(const std::vector<cv::Mat>& new_frames)
{
   int id = next_id++;

   if (new_frames.empty())
      throw std::runtime_error("Must have at least one frame");

   working_space[id] = std::make_shared<GpuWorkingSpace>(area(new_frames[0]), nD, range_max, compute_jacobian_on_gpu);
   for(auto& f : new_frames)
   {
      frames[f.data] = std::make_shared<GpuFrame>(f);
      frame_space[f.data] = id;      
   }

   return id;
}

void GpuFrameWarper::deregisterWorkingSpace(int id)
{
   working_space.erase(id);

   for(auto it = frame_space.begin(); it != frame_space.end();)
      if (it->second == id)
      {
         frames.erase(it->first);
         frame_space.erase(it++);         
      }
      else
         it++;
}

void GpuFrameWarper::setupReferenceInformation()
{
   size_t free_mem, total_mem;
   checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
   std::cout << "GPU Memory (total/free) : " << (total_mem / (1024 * 1024)) << " / " << (free_mem / (1024 * 1024)) << " Mb\n";

   size_t required_for_jacobian = 15 * 4 * area(reference);
   compute_jacobian_on_gpu = (free_mem > required_for_jacobian);

   if (compute_jacobian_on_gpu)
      std::cout << "Computing Jacobian on GPU\n";
   else
      std::cout << "Computing Jacobian on CPU (needed " << required_for_jacobian / (1024 * 1024) << " Mb)\n";

   float3 offset;
   double stack_duration = dims[Z] * image_params.frame_duration;
   offset.x = image_params.pixel_duration / stack_duration;
   offset.y = image_params.interline_duration / stack_duration;
   offset.z = image_params.frame_duration / stack_duration;

   range_max = std::max_element(VI_dW_dp.begin(), VI_dW_dp.end(), [](auto& a, auto& b) { return a.size() < b.size(); })->size();

   gpu_reference = std::make_unique<GpuReferenceInformation>(reference, offset, nD, range_max, compute_jacobian_on_gpu);

   if (compute_jacobian_on_gpu)
   {
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


}

std::shared_ptr<GpuFrame> GpuFrameWarper::getRegisteredFrame(const cv::Mat& frame)
{
   if (frames.count(frame.data) == 0)
      throw std::runtime_error("Unrecognised frame");
   return frames[frame.data];
}

std::shared_ptr<GpuWorkingSpace> GpuFrameWarper::getWorkingSpace(const cv::Mat& frame)
{
   if (frame_space.count(frame.data) == 0)
      throw std::runtime_error("Unrecognised frame");
   return working_space[frame_space[frame.data]];   
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
   auto w = getWorkingSpace(frame);   
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(w->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   return computeError(f.get(), w.get(), gpu_reference.get());
}

void GpuFrameWarper::getJacobian(const cv::Mat& frame, const std::vector<cv::Point3d>& D, column_vector& jac)
{
   auto f = getRegisteredFrame(frame);
   auto w = getWorkingSpace(frame);
   //auto Df = D2float3(D); => same as before

   if (compute_jacobian_on_gpu)
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
   else
   {
      cv::Mat error_image(dims, CV_32F);
      checkCudaErrors(cudaMemcpy(error_image.data, w->error_image, dims[X] * dims[Y] * dims[Z] * sizeof(float), cudaMemcpyDeviceToHost));
      computeJacobian(error_image, jac);   
   }

   /*
   std::cout << std::scientific << std::setw(5) << std::setprecision(3) << std::showpos;
   for(int i=0; i<jac.size(); i++)
   {
      if (i % 3 == 0) std::cout << "\n     > ";
      std::cout << jac(i) << "  |  ";
   }
   std::cout << "\n";
   */
}


void GpuFrameWarper::warpImage(const cv::Mat& frame, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value)
{      
   auto f = getRegisteredFrame(frame);
   auto w = getWorkingSpace(frame);   
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(w->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   computeWarp(f.get(), w.get(), gpu_reference.get());

   wimg = cv::Mat(dims, CV_32F);
   checkCudaErrors(cudaMemcpy(wimg.data, w->error_image, dims[X]*dims[Y]*dims[Z]*sizeof(float), cudaMemcpyDeviceToHost));   
}

void GpuFrameWarper::warpImageIntensityPreserving(const cv::Mat& frame, cv::Mat& wimg, cv::Mat& coverage, const std::vector<cv::Point3d>& D)
{
   auto f = getRegisteredFrame(frame);
   auto w = getWorkingSpace(frame);   
   auto Df = D2float3(D);

   checkCudaErrors(cudaMemcpy(w->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   computeIntensityPreservingWarp(f.get(), w.get(), gpu_reference.get());

   wimg = cv::Mat(dims, CV_32F);   
   checkCudaErrors(cudaMemcpy(wimg.data, w->error_image, dims[X]*dims[Y]*dims[Z]*sizeof(float), cudaMemcpyDeviceToHost));   

   coverage = cv::Mat(dims, CV_16U);   
   checkCudaErrors(cudaMemcpy(coverage.data, w->mask, dims[X]*dims[Y]*dims[Z]*sizeof(uint16_t), cudaMemcpyDeviceToHost));   
}

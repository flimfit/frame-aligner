#include "GpuFrameWarper.h"
#include "GpuFrameWarperKernels.h"

// Includes CUDA
#include <cuda_runtime.h>

#include "helper_cuda.h"


GpuFrameWarper::GpuFrameWarper()
{
   if (!getGpuSupportInformation())
      throw std::runtime_error("No supported GPU");
}

GpuSupportInformation GpuFrameWarper::getGpuSupportInformation()
{
   GpuSupportInformation support_information(true);
   
   int dev;
   cudaError_t code = cudaGetDevice(&dev);
   if (code != cudaSuccess)
   {
      if (code == cudaErrorInsufficientDriver)
      {
         support_information.message = "Insufficient Driver";
         support_information.remedy_message = "<p>A CUDA compatible card was found, but your installed driver is out of date.</p><p>To use GPU based realignment, please update your Nvidia driver at:<br/> <a href='http://www.nvidia.com/Download/Scan.aspx'>http://www.nvidia.com/Download/Scan.aspx</a></p>";
      }
      else if (code == cudaErrorNoDevice)
      {
         support_information.message = "No compatible GPU";
      }
      std::cout << "Could not load CUDA: " << _cudaGetErrorEnum(code) << ", will fall back on CPU\n";
      support_information.supported = false;
   }

   if (!checkCudaCapabilities(3, 0))
   {
      support_information.supported = false;
      support_information.message = "No compatible GPU";
   }
   return support_information;
}

void GpuFrameWarper::registerFrame(const cv::Mat& frame)
{
   std::unique_lock<std::mutex> lk(mutex);
   cv.wait(lk, [&]{ return (frames.size() < max_threads); });  // make sure we don't allocate too many frames   

   try
   {
      frames[frame.data] = frame_pool.get();      
      frames[frame.data]->set(frame);
   }
   catch(std::runtime_error e)
   {
      std::cout << "Could not allocate frame: " << e.what() << "\n";
      if (max_threads > 1)
         max_threads--; // we obviously don't have enough memory...
      lk.unlock();
      registerFrame(frame);
   }
}
 
void GpuFrameWarper::deregisterFrame(const cv::Mat& frame)
{
   {
      std::lock_guard<std::mutex> lk(mutex);   
      frames.erase(frame.data);   
   }
   cv.notify_all();
}

void GpuFrameWarper::setupReferenceInformation()
{
   // Setup reference information 

   stream_VI = true;
      
   auto p = processScanParameters(image_params);


   float3 offset;
   offset.x = p.pixel_duration / p.stack_duration;
   offset.y = p.interline_duration / p.stack_duration;
   offset.z = p.interframe_duration / p.stack_duration;

   range_max = std::max_element(VI_dW_dp.begin(), VI_dW_dp.end(), [](auto& a, auto& b) { return a.size() < b.size(); })->size();

   gpu_reference = std::make_unique<GpuReferenceInformation>(reference, offset, nD, range_max, stream_VI);
   gpu_reference->bidirectional = image_params.bidirectional;
   
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


   // Determine how many threads we can support
   size_t free_mem, total_mem;
   checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
   std::cout << "GPU Memory (total/free) : " << (total_mem / (1024 * 1024)) << " / " << (free_mem / (1024 * 1024)) << " Mb\n";

   size_t volume = dims[X] * dims[Y] * dims[Z];

   size_t working_space_size = volume / nD * sizeof(float) * 3 * 2 +     // approximate size of VI_dW_dp storage 
   dims[X] * dims[Y] * sizeof(float) * 2; // storage for warp
   size_t frame_size = volume * sizeof(float);

   max_threads = (free_mem - 32 * 1024 * 1024) / (frame_size + working_space_size); // leave a bit of free space
   max_threads = std::min(max_threads, 16); // only 16 texure refs currently

   if (max_threads <= 0)
      throw std::runtime_error("Not enough video memory to use GPU");

   std::cout << "  > Enough GPU memory for " << max_threads << " threads\n";
   
   // Setup working space

   GpuWorkingSpaceParams params;
   params.nxny = reference.size[X] * reference.size[Y];
   params.nD = nD;
   params.range_max = range_max;
   pool.setInit(params);

   
   frame_pool.setInit({dims[X], dims[Y], dims[Z]});
    
   /*
   if (!stream_VI)
   {
      checkCudaErrors(cudaMemcpy(gpu_reference->VI_dW_dp, VI_dW_dp_host, range_max * nD * sizeof(float3), cudaMemcpyHostToDevice));      
      checkCudaErrors(cudaFreeHost(VI_dW_dp_host));
      VI_dW_dp_host = nullptr;
   }
   */

}

void GpuFrameWarper::clearTemp()
{
   frame_pool.clear();
   pool.clear();
}

GpuFrame* GpuFrameWarper::getRegisteredFrame(const cv::Mat& frame)
{
   std::lock_guard<std::mutex> lk(mutex);   
   if (frames.count(frame.data) == 0)
      throw std::runtime_error("Unrecognised frame");
   return frames[frame.data].get();
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
   return computeError(f, w.get(), gpu_reference.get());
}

void GpuFrameWarper::getJacobian(const cv::Mat& frame, const std::vector<cv::Point3d>& D, column_vector& jac)
{
   auto f = getRegisteredFrame(frame);
   auto w = pool.get();
   //auto Df = D2float3(D); => same as before

   //if (compute_jacobian_on_gpu)
   {
      std::vector<float3> jacv = computeJacobianGpu(f, w.get(), gpu_reference.get());
      
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

   wimg = cv::Mat(dims, CV_32F);

   checkCudaErrors(cudaMemcpy(w->D, Df.data(), nD*sizeof(float3), cudaMemcpyHostToDevice));
   computeWarp(f, w.get(), gpu_reference.get(), (float*) wimg.data);
}

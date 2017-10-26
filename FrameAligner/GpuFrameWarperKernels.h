
#pragma once

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template<typename T>
class CudaMemory
{
public:
   ~CudaMemory() { if (data) cudaFree(data); }
   operator T*() { return data; }
   
   void allocate(size_t size)
   {
      if (data) cudaFree(data);
      auto success = cudaMalloc((void**) &data, size * sizeof(T));     
      if (success != cudaSuccess)
         throw std::runtime_error("Not enough memory to allocate");
   }
   
private:
   T* data = nullptr;
};

template<typename T>
class CudaHostMemory
{
public:
   ~CudaHostMemory() { if (data) cudaFreeHost(data); }
   operator T*() { return data; }
   
   void allocate(size_t size)
   {
      if (data) cudaFree(data);
      auto success = cudaMallocHost((void**) &data, size * sizeof(T));     
      if (success != cudaSuccess)
         throw std::runtime_error("Not enough memory to allocate");
   }

   
private:
   T* data = nullptr;
};


class GpuFrame
{
public:
   GpuFrame(int3 size);
   ~GpuFrame();

   void set(const cv::Mat& frame);

   bool isSame(const cv::Mat& frame_) const { return (frame.data == frame_.data); }

   int getTextureId() { return texture; };

   int3 size;
   cv::Mat frame;
   
protected:
   cudaArray *cu_array;
   int texture;
};

struct GpuWorkingSpaceParams
{
   int nxny;
   int nD;
   int range_max;
};

class GpuWorkingSpace
{
public:
   GpuWorkingSpace(GpuWorkingSpaceParams params);
   ~GpuWorkingSpace();


   CudaMemory<float3> VI_dW_dp;   
   CudaMemory<float> error_image;
   CudaMemory<float> error_sum;
   CudaMemory<uint16_t> mask;
   CudaMemory<float3> D;
   int3 size;
   std::vector<cudaStream_t> stream;
   CudaHostMemory<float> host_buffer;
};

struct GpuRange
{
   int begin;
   int end;
};

class GpuReferenceInformation
{
public:

   GpuReferenceInformation(const cv::Mat& reference, float3 offset, int nD, int range_max, bool compute_jacobian_on_gpu);

   cv::Mat cvref;
   CudaMemory<float> reference;
   CudaHostMemory<float3> VI_dW_dp_host;
   std::vector<GpuRange> range;
   float3 offset;
   int nD = 0;
   int range_max = 0;
   bool stream_VI = false;
};

struct WarpParams
{
   int tex_id;
   int3 size;
   float3 offset;
   float* reference;
   int nD;
   float3* D;
};

void computeWarp(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref, float* warp);
//void computeIntensityPreservingWarp(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref);
double computeError(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref);
std::vector<float3> computeJacobianGpu(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref);

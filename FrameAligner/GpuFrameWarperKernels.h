
#pragma once

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

class GpuFrame
{
public:
   GpuFrame(const cv::Mat& frame);
   ~GpuFrame();

   bool isSame(const cv::Mat& frame_) const { return (frame.data == frame_.data); }

   int getTextureId() { return texture; };

   int3 size;
   cv::Mat frame;
   
protected:
   cudaArray *cu_array;
   int texture;
};

class GpuWorkingSpace
{
public:
   GpuWorkingSpace(int volume, int nD, int range_max, bool calculate_jacobian_on_gpu);
   ~GpuWorkingSpace();

   float* error_image;
   float* error_sq_image;
   float* error_sum;
   uint16_t* mask;
   float3* jacobian;
   float3* D;
   int3 size;
   
protected:
   bool calculate_jacobian_on_gpu;
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
   ~GpuReferenceInformation();

   cv::Mat cvref;
   float* reference = nullptr;
   float3 *VI_dW_dp = nullptr;
   GpuRange *range = nullptr;
   float3 offset;
   int nD = 0;
   int range_max = 0;
   bool compute_jacobian_on_gpu = false;
};

void computeWarp(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref);
void computeIntensityPreservingWarp(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref);
double computeError(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref);
std::vector<float3> computeJacobianGpu(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref);

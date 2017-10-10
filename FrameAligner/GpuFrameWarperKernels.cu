#include "GpuFrameWarperKernels.h"
#include "GpuTextureManager.h"


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "helper_cuda.h"

texture<float, 3, cudaReadModeElementType> tex0;
texture<float, 3, cudaReadModeElementType> tex1;
texture<float, 3, cudaReadModeElementType> tex2;
texture<float, 3, cudaReadModeElementType> tex3;

GpuTextureManager* GpuTextureManager::gpu_texture_manager = nullptr;

texture<float, 3, cudaReadModeElementType>& getTexture(int id)
{
   switch (id)
   {
   case 0: return tex0;
   case 1: return tex1;
   case 2: return tex2;
   case 3: return tex3;
   }
   throw std::runtime_error("Invalid texture reference");
}

GpuTextureManager* GpuTextureManager::instance()
{
   if (!gpu_texture_manager)
      gpu_texture_manager = new GpuTextureManager;
   return gpu_texture_manager;
}

int GpuTextureManager::getTextureId()
{
   std::unique_lock<std::mutex> lk(mutex);
   cv.wait(lk, [this] { return !free_textures.empty(); });
   int texture = free_textures.back();

   free_textures.pop_back();
   return texture;
}

void GpuTextureManager::returnTextureId(int t)
{
   {
      std::lock_guard<std::mutex> lk(mutex);
      free_textures.push_back(t);
   }
   cv.notify_one();
}

GpuTextureManager::GpuTextureManager()
{
   for (int i = 0; i < 16; i++)
      free_textures.push_back(i);
}


class A
{
public:
   int a;
   int b;

   __device__ A& operator=(int other)
   {
      a = other;
      b = other;
      return *this;
   }

   __device__ A& operator+=(const A& other)
   {
      a += other.a;
      b += other.b;
      return *this;
   }
};

template <unsigned int blockSize, class T>
__device__ void warpReduce(T *sdata, unsigned int tid) {
   if (blockSize >= 64) { sdata[tid] += sdata[tid + 32]; }
   if (blockSize >= 32) { sdata[tid] += sdata[tid + 16]; }
   if (blockSize >= 16) { sdata[tid] += sdata[tid + 8]; }
   if (blockSize >= 8) { sdata[tid] += sdata[tid + 4]; }
   if (blockSize >= 4) { sdata[tid] += sdata[tid + 2]; }
   if (blockSize >= 2) { sdata[tid] += sdata[tid + 1]; }
}

template <unsigned int blockSize, class T>
__global__ void reduceSum(T *g_idata, T *g_odata, unsigned int n) {
   extern __shared__ char sdata_[];
   T* sdata = reinterpret_cast<T*>(sdata_);

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*(blockSize * 2) + tid;
   unsigned int gridSize = blockSize * 2 * gridDim.x;
   sdata[tid] = 0;
   while (i < n) 
   {
      sdata[tid] += g_idata[i];
      sdata[tid] += g_idata[i + blockSize]; 
      i += gridSize;
   }
   __syncthreads();
   if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
   if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
   if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
   if (tid < 32) warpReduce<blockSize>(sdata, tid);
   if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; };
}



__device__ float3 warpPoint(int nD, float3* D, float3 offset, int x, int y, int z)
{
   float Didx = (x * offset.x + y * offset.y + z * offset.z) * (nD-1);
   int idx = Didx; // => floor
   float f = Didx - idx;
   float3 p;
   p.x = f * D[idx + 1].x + (1 - f) * D[idx].x;
   p.y = f * D[idx + 1].y + (1 - f) * D[idx].y;
   p.z = f * D[idx + 1].z + (1 - f) * D[idx].z;
   return p;
}


__global__ void warpAndGetError(int tex_id, int3 size, float3 offset, float* reference, float* error_img, int nD, float3* D)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;   
   int z = threadIdx.z + blockDim.z * blockIdx.z;
   int idx = x + y * size.x + z * (size.x * size.y);

   float3 p = warpPoint(nD, D, offset, x, y, z);

   float v = 0;
   if (tex_id == 0)
      v = tex3D(tex0, p.x, p.y, p.z);
   else if (tex_id == 1)
      v = tex3D(tex1, p.x, p.y, p.z);
   else if (tex_id == 2)
      v = tex3D(tex2, p.x, p.y, p.z);
   else if (tex_id == 3)
      v = tex3D(tex3, p.x, p.y, p.z);


   float censor = v > 0; // set out of range values to zero

   // we have added 1 to use zero as special case
   error_img[idx] = censor * (v - 1.0 - reference[idx]); 
}





__global__ void computeJacobian(float* error_img, float* jac, float3* VI_dW_dp, int nD, int n_dim, int range_max, GpuRange* range)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;

   for (int i = 1; i < nD; i++)
   {
      int p0 = range[i - 1].begin;
      int p1 = range[i - 1].end;
      int p = x + p0;

      if (p <= p1)
      {
         jac[i*n_dim] += VI_dW_dp[i*range_max + p].x * error_img[p]; // x 
         jac[i*n_dim + 1] += VI_dW_dp[i*range_max + p].y * error_img[p]; // y
         if (n_dim == 3)
            jac[i*n_dim + 2] += VI_dW_dp[i*range_max + p].z * error_img[p]; // z        
      }
   }
   for (int i = 0; i < (nD - 1); i++)
   {
      int p0 = range[i].begin;
      int p1 = range[i].end;
      int p = x + p0;

      if (p <= p1)
         jac[i*n_dim] += VI_dW_dp[i*range_max + p].x * error_img[p]; // x
      jac[i*n_dim + 1] += VI_dW_dp[i*range_max + p].y * error_img[p]; // y
      if (n_dim == 3)
         jac[i*n_dim + 2] += VI_dW_dp[i*range_max + p].z * error_img[p]; // z

   }


   /*
   for (int i = 1; i < nD; i++)
   {
      int p0 = D_range[i - 1].begin;
      int p1 = D_range[i - 1].end;
      for (int p = p0; p < p1; p++)
      {
         jac[i*n_dim] += VI_dW_dp_x[i][p] * err_ptr[p]; // x 
         jac[i*n_dim + 1] += VI_dW_dp_y[i][p] * err_ptr[p]; // y
         if (n_dim == 3)
            jac(i*n_dim + 2) += VI_dW_dp_z[i][p] * err_ptr[p]; // z        
      }
   }
   for (int i = 0; i < (nD - 1); i++)
   {
      int p0 = D_range[i].begin;
      int p1 = D_range[i].end;
      for (int p = p0; p < p1; p++)
      {
         jac(i*n_dim) += VI_dW_dp_x[i][p] * err_ptr[p]; // x
         jac(i*n_dim + 1) += VI_dW_dp_y[i][p] * err_ptr[p]; // y
         if (n_dim == 3)
            jac(i*n_dim + 2) += VI_dW_dp_z[i][p] * err_ptr[p]; // z
         
      }
   }
   */
   
}


GpuFrame::GpuFrame(cv::Mat frame, int nD)
{
   auto tex_manager = GpuTextureManager::instance();
   texture = tex_manager->getTextureId();
   
   auto& tex = getTexture(texture);
  
   // Set texture parameters
   tex.addressMode[0] = cudaAddressModeBorder;
   tex.addressMode[1] = cudaAddressModeBorder;
   tex.filterMode = cudaFilterModeLinear;
   tex.normalized = false;

   // Add 1 to frame value -> we want to use zero as a special case
   cv::Mat frame_cpy;
   frame.copyTo(frame_cpy);
   frame_cpy += 1.0f;

   size.x = frame.size[2];
   size.y = frame.size[1];
   size.z = frame.size[0];
   size_t volume = size.x * size.y * size.z;

   // Allocate array and copy image data
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
   cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);
   size_t copy_size = volume * sizeof(float);
   checkCudaErrors(cudaMalloc3DArray(&cu_array, &channelDesc, extent));
   checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, frame.data, copy_size, cudaMemcpyHostToDevice));

   // Bind the array to the texture
   checkCudaErrors(cudaBindTextureToArray(&tex, cu_array, &channelDesc));


   checkCudaErrors(cudaMalloc(&error_sum, 1));
   checkCudaErrors(cudaMalloc(&error_image, volume));
   checkCudaErrors(cudaMalloc(&D, nD));
}

GpuFrame::~GpuFrame()
{
   checkCudaErrors(cudaFree(&cu_array));

   auto tex_manager = GpuTextureManager::instance();
   texture = tex_manager->getTextureId();
   auto& tex = getTexture(texture);
   checkCudaErrors(cudaUnbindTexture(tex));

   checkCudaErrors(cudaFree(error_sum));
   checkCudaErrors(cudaFree(error_image));
   checkCudaErrors(cudaFree(D));
}


GpuReferenceInformation::GpuReferenceInformation(const cv::Mat& ref_, float3 offset, int nD, int range_max) :
   offset(offset), nD(nD), range_max(range_max)
{
   int n_px = ref_.size[0] * ref_.size[1] * ref_.size[2];

   checkCudaErrors(cudaMalloc(&reference, n_px));
   checkCudaErrors(cudaMalloc(&VI_dW_dp, range_max * nD));
   checkCudaErrors(cudaMalloc(&range, nD));

   checkCudaErrors(cudaMemcpy(reference, ref_.data, n_px * sizeof(float), cudaMemcpyHostToDevice));
}

GpuReferenceInformation::~GpuReferenceInformation()
{
   checkCudaErrors(cudaFree(VI_dW_dp));
   checkCudaErrors(cudaFree(range));
   checkCudaErrors(cudaFree(reference));
}



double computeError(GpuFrame* frame, GpuReferenceInformation* gpu_ref)
{
   auto size = frame->size;
   int volume = size.x * size.y * size.z;

   dim3 dimBlock(32, 32, 1);
   dim3 dimGrid(size.x / 32, size.y / 32, size.z);

   int id = frame->getTextureId();

   warpAndGetError<<<dimGrid, dimBlock, 0>>>(id, frame->size, gpu_ref->offset, gpu_ref->reference, frame->error_image, gpu_ref->nD, frame->D);

   const int block_size = 512;
   int n_block = volume / block_size;
   reduceSum<block_size> << <n_block, block_size, block_size * sizeof(float) >> > (frame->error_image, frame->error_sum, volume);

   float error_sum;
   cudaMemcpy(&error_sum, frame->error_sum, sizeof(float), cudaMemcpyDeviceToHost);

   return error_sum;
}

void computeJacobian(GpuFrame* frame, GpuReferenceInformation* gpu_ref)
{
   //float* error_img, float* jac, float3* VI_dW_dp, int nD, int n_dim, int range_max, GpuRange* range
}
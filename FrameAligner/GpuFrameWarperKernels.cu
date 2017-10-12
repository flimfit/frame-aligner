#include "GpuFrameWarperKernels.h"
#include "GpuTextureManager.h"


#include <cuda.h>
#include <cuda_runtime_api.h>
#include "helper_cuda.h"

const int num_tex = 16;
texture<float, 3, cudaReadModeElementType> tex0;
texture<float, 3, cudaReadModeElementType> tex1;
texture<float, 3, cudaReadModeElementType> tex2;
texture<float, 3, cudaReadModeElementType> tex3;
texture<float, 3, cudaReadModeElementType> tex4;
texture<float, 3, cudaReadModeElementType> tex5;
texture<float, 3, cudaReadModeElementType> tex6;
texture<float, 3, cudaReadModeElementType> tex7;
texture<float, 3, cudaReadModeElementType> tex8;
texture<float, 3, cudaReadModeElementType> tex9;
texture<float, 3, cudaReadModeElementType> tex10;
texture<float, 3, cudaReadModeElementType> tex11;
texture<float, 3, cudaReadModeElementType> tex12;
texture<float, 3, cudaReadModeElementType> tex13;
texture<float, 3, cudaReadModeElementType> tex14;
texture<float, 3, cudaReadModeElementType> tex15;

float3 operator+(const float3& a, const float3& b)
{
   float3 c;
   c.x = a.x + b.x;
   c.y = a.y + b.y;
   c.z = a.z + b.z;
   return c;
}

__device__ float3& operator+=(float3& a, const float3& b)
{
   a.x += b.x;
   a.y += b.y;
   a.z += b.z;
   return a;
}

__device__ float3 operator-(const float3& a, const float3& b)
{
   float3 c;
   c.x = a.x - b.x;
   c.y = a.y - b.y;
   c.z = a.z - b.z;
   return c;
}

__device__ float3& operator-=(float3& a, const float3& b)
{
   a.x -= b.x;
   a.y -= b.y;
   a.z -= b.z;
   return a;
}

class f3
{
public:

   __device__ f3& operator=(float other)
   {
      x = other;
      y = other;
      z = other;
      return *this;
   }

   __device__ f3& operator=(f3& other)
   {
      x = other.x;
      y = other.y;
      z = other.z;
      return *this;
   }

   __device__ f3& operator+=(const f3& other)
   {
      x += other.x;
      y += other.y;
      z += other.z;
      return *this;
   }

   float x = 0;
   float y = 0;
   float z = 0;
};

GpuTextureManager* GpuTextureManager::gpu_texture_manager = nullptr;

texture<float, 3, cudaReadModeElementType>& getTexture(int id)
{
   switch (id)
   {
   case 0: return tex0;
   case 1: return tex1;
   case 2: return tex2;
   case 3: return tex3;
   case 4: return tex4;
   case 5: return tex5;
   case 6: return tex6;
   case 7: return tex7;
   case 8: return tex8;
   case 9: return tex9;
   case 10: return tex10;
   case 11: return tex11;
   case 12: return tex12;
   case 13: return tex13;
   case 14: return tex14;
   case 15: return tex15;
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
   for (int i = 0; i < num_tex; i++)
      free_textures.push_back(i);
}


template <unsigned int blockSize, class T>
__device__ void warpReduce(T *sdata, unsigned int tid) {

}

template <unsigned int blockSize, class T>
__global__ void reduceSum(T *g_idata, T *g_odata, unsigned int n) {
   extern __shared__ char sdata_[];
   T* sdata = reinterpret_cast<T*>(sdata_);

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * 2 * blockSize + tid;
   unsigned int gridSize = blockSize * 2 * gridDim.x;
   sdata[tid] = 0.0f;
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
   if (blockSize >= 64) { if (tid < 32) { sdata[tid] += sdata[tid + 32]; } __syncthreads(); }
   if (blockSize >= 32) { if (tid < 16) { sdata[tid] += sdata[tid + 16]; } __syncthreads(); }
   if (blockSize >= 16) { if (tid < 8) { sdata[tid] += sdata[tid + 8]; } __syncthreads(); }
   if (blockSize >= 8) { if (tid < 4) { sdata[tid] += sdata[tid + 4]; } __syncthreads(); }
   if (blockSize >= 4) { if (tid < 2) { sdata[tid] += sdata[tid + 2]; } __syncthreads(); }
   if (blockSize >= 2) { if (tid < 1) { sdata[tid] += sdata[tid + 1]; } __syncthreads(); }
   if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; };
}



__device__ float3 warpPoint(int nD, const float3* D, float3 offset, int x, int y, int z)
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

__device__ float getPoint(int tex_id, float3 p)
{
   p.x += 0.5f;
   p.y += 0.5f;
   p.z += 0.5f;
   
   switch(tex_id)
   {
   case 0: return tex3D(tex0, p.x, p.y, p.z);
   case 1: return tex3D(tex1, p.x, p.y, p.z);
   case 2: return tex3D(tex2, p.x, p.y, p.z);
   case 3: return tex3D(tex3, p.x, p.y, p.z);
   case 4: return tex3D(tex4, p.x, p.y, p.z);
   case 5: return tex3D(tex5, p.x, p.y, p.z);
   case 6: return tex3D(tex6, p.x, p.y, p.z);
   case 7: return tex3D(tex7, p.x, p.y, p.z);
   case 8: return tex3D(tex8, p.x, p.y, p.z);
   case 9: return tex3D(tex9, p.x, p.y, p.z);
   case 10: return tex3D(tex10, p.x, p.y, p.z);
   case 11: return tex3D(tex11, p.x, p.y, p.z);
   case 12: return tex3D(tex12, p.x, p.y, p.z);
   case 13: return tex3D(tex13, p.x, p.y, p.z);
   case 14: return tex3D(tex14, p.x, p.y, p.z);
   case 15: return tex3D(tex15, p.x, p.y, p.z);   
   }  
   return 0;
}


__global__ void warp(int tex_id, int3 size, float3 offset, float* warp_img, int nD, float3* D)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;   
   int z = threadIdx.z + blockDim.z * blockIdx.z;
   int idx = x + y * size.x + z * (size.x * size.y);

   float3 p = warpPoint(nD, D, offset, x, y, z);
   p.x += x; p.y += y; p.z += z;
   float v = getPoint(tex_id, p);

   warp_img[idx] = (v != 0.0f) * (v - 1.0f);
}

__global__ void warpAndGetError(int tex_id, int3 size, float3 offset, float* __restrict__ reference, float* __restrict__ error_img, float* __restrict__ error_sq_img, int nD, float3* D)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;   
   int z = threadIdx.z + blockDim.z * blockIdx.z;
   int idx = x + y * size.x + z * (size.x * size.y);

   float3 p = warpPoint(nD, D, offset, x, y, z);
   p.x += x; p.y += y; p.z += z;
   float v = getPoint(tex_id, p);

   float mask = (v != 0.0f) ? 1.0f : 0.0f; // set out of range values to zero
   v -= (1.0f + reference[idx]); // we have added 1 to use zero as special case
   error_img[idx] = mask * v; 
   error_sq_img[idx] = mask * v * v; 
}

__global__ void warpIntensityPreserving(int tex_id, int3 size, float3 offset, float* warp_img, uint16_t* mask_img, int nD, float3* D)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;   
   int z = threadIdx.z + blockDim.z * blockIdx.z;

   float3 p0;
   p0.x = x; p0.y = y; p0.z = z;
   float v = getPoint(tex_id, p0);

   float3 p = warpPoint(nD, D, offset, x, y, z);
   p0 -= p;

   x = round(p0.x);
   y = round(p0.y);
   z = round(p0.z);
   
   uint16_t valid = (x >= 0) && (x < size.x) && (y >= 0) && (y < size.y) && (z >= 0) && (z < size.z); 
   int idx = x + y * size.x + z * (size.x * size.y);
   
   if (valid)
   {
      warp_img[idx] += (v != 0.0f) * (v - 1.0f);
      mask_img[idx]++;   
   }
}



__global__ void computeJacobian(float* error_img, float3* __restrict__ jac_, float3* __restrict__ VI_dW_dp_, int nD, int range_max, GpuRange* range)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   float3* jac = jac_ + x;
   float3* VI_dW_dp = VI_dW_dp_ + x;

   if (x < range_max)
      for (int i = 0; i < nD; i++)
      {
         int p0 = range[i].begin;
         int p1 = range[i].end;
         int p = x + p0;
         int idx = range_max*i;

         jac[idx].x = 0;
         jac[idx].y = 0;
         jac[idx].z = 0;   
         
         if (p <= p1)
         {
            float ep = error_img[p];
            jac[idx].x = VI_dW_dp[idx].x * ep;
            jac[idx].y = VI_dW_dp[idx].y * ep;
            jac[idx].z = VI_dW_dp[idx].z * ep;  
         }
      }
}


GpuFrame::GpuFrame(const cv::Mat& frame_)
{
   auto tex_manager = GpuTextureManager::instance();
   texture = tex_manager->getTextureId();
   
   auto& tex = getTexture(texture);
 
   frame = frame_;

   // Set texture parameters
   tex.addressMode[0] = cudaAddressModeBorder;
   tex.addressMode[1] = cudaAddressModeBorder;
   tex.addressMode[2] = cudaAddressModeBorder;
   tex.filterMode = cudaFilterModePoint;
   tex.normalized = false;

   // Add 1 to frame value -> we want to use zero as a special case
   cv::Mat frame_cpy;
   frame.copyTo(frame_cpy);
   frame_cpy += 1.0f;

   size.x = frame.size[2];
   size.y = frame.size[1];
   size.z = frame.size[0];

   // Allocate array and copy image data
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);
   checkCudaErrors(cudaMalloc3DArray(&cu_array, &channelDesc, extent));
   
   cudaMemcpy3DParms copy_params = {0};
   copy_params.srcPtr = make_cudaPitchedPtr((void*) frame_cpy.data, size.x * sizeof(float), size.x, size.y);
   copy_params.dstArray = cu_array;
   copy_params.extent = extent;
   copy_params.kind = cudaMemcpyHostToDevice;
   checkCudaErrors(cudaMemcpy3D(&copy_params));
   
   // Bind the array to the texture
   checkCudaErrors(cudaBindTextureToArray(&tex, cu_array, &channelDesc));
}


GpuFrame::~GpuFrame()
{
   auto& tex = getTexture(texture);
   checkCudaErrors(cudaUnbindTexture(tex));
   checkCudaErrors(cudaFreeArray(cu_array));

   auto tex_manager = GpuTextureManager::instance();
   tex_manager->returnTextureId(texture);
}


GpuWorkingSpace::GpuWorkingSpace(int volume, int nD, int range_max, bool calculate_jacobian_on_gpu) : 
   calculate_jacobian_on_gpu(calculate_jacobian_on_gpu)
{
   checkCudaErrors(cudaMalloc((void**) &error_sum, 1024 * sizeof(float)));
   checkCudaErrors(cudaMalloc((void**) &error_image, volume*sizeof(float)));
   checkCudaErrors(cudaMalloc((void**) &error_sq_image, volume*sizeof(float)));
   checkCudaErrors(cudaMalloc((void**) &mask, volume*sizeof(uint16_t)));
   checkCudaErrors(cudaMalloc((void**) &D, nD*sizeof(float3)));

   if (calculate_jacobian_on_gpu)   
      checkCudaErrors(cudaMalloc((void**) &jacobian, range_max * nD * sizeof(float3)));

   cudaStreamCreate(&stream);
}

GpuWorkingSpace::~GpuWorkingSpace()
{
   checkCudaErrors(cudaFree(error_sum));
   checkCudaErrors(cudaFree(error_image));
   checkCudaErrors(cudaFree(mask));
   checkCudaErrors(cudaFree(error_sq_image));
   checkCudaErrors(cudaFree(D));

   if (calculate_jacobian_on_gpu)
      checkCudaErrors(cudaFree(jacobian));

   cudaStreamDestroy(stream);
}


GpuReferenceInformation::GpuReferenceInformation(const cv::Mat& ref_, float3 offset, int nD, int range_max, bool compute_jacobian_on_gpu) :
   offset(offset), nD(nD), range_max(range_max), compute_jacobian_on_gpu(compute_jacobian_on_gpu)
{
   size_t n_px = ref_.size[0] * ref_.size[1] * ref_.size[2];

   cvref = ref_;

   if (compute_jacobian_on_gpu)
   {
      checkCudaErrors(cudaMalloc((void**) &VI_dW_dp, range_max * nD * sizeof(float3)));
      checkCudaErrors(cudaMalloc((void**) &range, nD * sizeof(GpuRange)));   
   }

   checkCudaErrors(cudaMalloc((void**) &reference, n_px * sizeof(float)));
   checkCudaErrors(cudaMemcpy(reference, ref_.data, n_px * sizeof(float), cudaMemcpyHostToDevice));
}

GpuReferenceInformation::~GpuReferenceInformation()
{
   if (compute_jacobian_on_gpu)
   {
      checkCudaErrors(cudaFree(VI_dW_dp));
      checkCudaErrors(cudaFree(range));   
   }
   checkCudaErrors(cudaFree(reference));
}


void computeWarp(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref)
{
   auto size = frame->size;
   dim3 dimBlock(32, 32, 1);
   dim3 dimGrid(size.x / 32, size.y / 32, size.z);

   int id = frame->getTextureId();
   warp<<<dimGrid, dimBlock, 0, w->stream>>>(id, frame->size, gpu_ref->offset, w->error_image, gpu_ref->nD, w->D);
   getLastCudaError("Kernel execution failed [ warp ]");
}


void computeIntensityPreservingWarp(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref)
{
   auto size = frame->size;
   dim3 dimBlock(32, 32, 1);
   dim3 dimGrid(size.x / 32, size.y / 32, size.z);

   cudaMemset(w->error_image, 0, size.x * size.y * size.z * sizeof(float));
   cudaMemset(w->mask, 0, size.x * size.y * size.z * sizeof(uint16_t));
   
   int id = frame->getTextureId();
   warpIntensityPreserving<<<dimGrid, dimBlock, 0, w->stream>>>(id, frame->size, gpu_ref->offset, w->error_image, w->mask, gpu_ref->nD, w->D);
   getLastCudaError("Kernel execution failed [ warpIntensityPreserving ]");
}


double computeError(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref)
{
   auto size = frame->size;
   int volume = size.x * size.y * size.z;

   dim3 dimBlock(32, 32, 1);
   dim3 dimGrid(size.x / 32, size.y / 32, size.z);

   int id = frame->getTextureId();
   warpAndGetError<<<dimGrid, dimBlock, 0, w->stream>>>(id, frame->size, gpu_ref->offset, gpu_ref->reference, w->error_image, w->error_sq_image, gpu_ref->nD, w->D);
   getLastCudaError("Kernel execution failed [ reduceSum ]");

   const int block_size = 512;
   int n_block = 1; //volume / (block_size * 2);
   reduceSum<block_size><<<dim3(1,1,1), dim3(block_size,1,1), block_size * sizeof(float), w->stream>>> (w->error_sq_image, w->error_sum, volume);
   getLastCudaError("Kernel execution failed [ reduceSum ]");

   std::vector<float> error_sum(n_block);
   checkCudaErrors(cudaMemcpy(error_sum.data(), w->error_sum, n_block*sizeof(float), cudaMemcpyDeviceToHost));

   for(int i=1; i<n_block; i++)
      error_sum[0] += error_sum[i];

   return error_sum[0];
}

std::vector<float3> computeJacobianGpu(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref)
{  
   int range_max = gpu_ref->range_max;

   int block_size = 512;
   int n_block = (range_max + block_size - 1) / block_size;

   int id = frame->getTextureId();
   computeJacobian<<<n_block, block_size, 0, w->stream>>>(w->error_image, w->jacobian,
      gpu_ref->VI_dW_dp, gpu_ref->nD, range_max, gpu_ref->range);
   
   getLastCudaError("Kernel execution failed [ computeJacobian ]");

   std::vector<float3> jac_out(gpu_ref->nD);

   for(int i=0; i<gpu_ref->nD; i++)
   {
      int n_block = 1;
      const int block_size = 512;
      reduceSum<block_size><<<n_block, block_size, block_size * sizeof(float3), w->stream>>> ((f3*)w->jacobian + i * range_max, (f3*)w->error_sum, range_max);
      getLastCudaError("Kernel execution failed [ reduceSum ]");   
      
      std::vector<float3> jac_sum(n_block);
      checkCudaErrors(cudaMemcpy(jac_sum.data(), w->error_sum, n_block*sizeof(float3), cudaMemcpyDeviceToHost));   
      jac_out[i] = jac_sum[0];
   }   

   return jac_out;
}
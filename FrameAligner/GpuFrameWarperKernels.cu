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

__host__ __device__ float3 operator+(const float3& a, const float3& b)
{
   float3 c;
   c.x = a.x + b.x;
   c.y = a.y + b.y;
   c.z = a.z + b.z;
   return c;
}

__host__ __device__ float3& operator+=(float3& a, const float3& b)
{
   a.x += b.x;
   a.y += b.y;
   a.z += b.z;
   return a;
}

__host__ __device__ float3 operator-(const float3& a, const float3& b)
{
   float3 c;
   c.x = a.x - b.x;
   c.y = a.y - b.y;
   c.z = a.z - b.z;
   return c;
}

__host__ __device__ float3& operator-=(float3& a, const float3& b)
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

   __device__ f3 operator*(float other)
   {
      f3 out;
      out.x = x * other;
      out.y = y * other;
      out.z = z * other;
      return out;
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

__device__ float warpAndGetError(WarpParams w, int idx)
{
   int x = idx % w.size.x;
   int iy = idx / w.size.x;
   int y = iy % w.size.y;
   int z = iy / w.size.y;

   float3 p = warpPoint(w.nD, w.D, w.offset, x, y, z);
   p.x += x; p.y += y; p.z += z;
   float v = getPoint(w.tex_id, p);

   float mask = (v != 0.0f) ? 1.0f : 0.0f; // set out of range values to zero
   v -= (1.0f + w.reference[idx]); // we have added 1 to use zero as special case
   return mask * v; 
}


template <unsigned int blockSize, class T>
__device__ void warpReduce(T *sdata, unsigned int tid) 
{
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
}

template <unsigned int blockSize>
__global__ void jacobianReduceSum(f3 * g_odata, WarpParams w, f3* __restrict__ VI_dW_dp, int n, GpuRange range) 
{
   extern __shared__ f3 sdata2[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockSize + tid;
   unsigned int gridSize = blockSize * gridDim.x;

   sdata2[tid] = 0.0f;
   while (i < n) 
   {
      sdata2[tid] += (VI_dW_dp[i] * warpAndGetError(w, i + range.begin));
      i += gridSize;
   }
   warpReduce<blockSize>(sdata2, tid);
   if (tid == 0) { g_odata[blockIdx.x] = sdata2[0]; };
}

template <unsigned int blockSize>
__global__ void reduceSumSquared(WarpParams w, float *g_odata, unsigned int n) 
{
   extern __shared__ float sdata1[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x * blockSize + tid;
   unsigned int gridSize = blockSize * gridDim.x;
   sdata1[tid] = 0.0f;
   while (i < n) 
   {
      float e1 = warpAndGetError(w, i);
      //float e2 = warpAndGetError(w, i + blockSize);
      sdata1[tid] += (e1 * e1); 
      i += gridSize;
   }
   warpReduce<blockSize>(sdata1, tid);
   if (tid == 0) { g_odata[blockIdx.x] = sdata1[0]; };
}



__global__ void warp(WarpParams w, float* warp_img, int z)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;   
   int idx = x + y * w.size.x;

   float3 p = warpPoint(w.nD, w.D, w.offset, x, y, z);
   p.x += x; p.y += y; p.z += z;
   float v = getPoint(w.tex_id, p);

   warp_img[idx] = (v != 0.0f) * (v - 1.0f);
}



__global__ void warpAndGetError(int tex_id, int3 size, float3 offset, float* __restrict__ reference, float* __restrict__ error_img, int nD, float3* D)
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


GpuFrame::GpuFrame(const cv::Mat& frame_)
{
   // Get texture reference
   auto tex_manager = GpuTextureManager::instance();
   texture = tex_manager->getTextureId();

   // Try and allocate array
   size.x = frame_.size[2];
   size.y = frame_.size[1];
   size.z = frame_.size[0];

   // Allocate array and copy image data
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   cudaExtent extent = make_cudaExtent(size.x, size.y, size.z);
   int success = cudaMalloc3DArray(&cu_array, &channelDesc, extent);

   // If we couldn't get enough memory throw an exception
   if (success != cudaSuccess)
   {
      tex_manager->returnTextureId(texture);
      std::cout << "Not enough memory to allocate frame!\n"; 
      throw std::runtime_error("Not enough memory to allocate frame");
   }
   

   auto& tex = getTexture(texture);
 
   size_t free_mem, total_mem;
   checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
   std::cout << "[Allocating frame] " << (total_mem / (1024 * 1024)) << " / " << (free_mem / (1024 * 1024)) << " Mb\n";

   frame = frame_;

   // Set texture parameters
   tex.addressMode[0] = cudaAddressModeBorder;
   tex.addressMode[1] = cudaAddressModeBorder;
   tex.addressMode[2] = cudaAddressModeBorder;
   tex.filterMode = cudaFilterModeLinear;
   tex.normalized = false;

   // Add 1 to frame value -> we want to use zero as a special case
   cv::Mat frame_cpy;
   frame.copyTo(frame_cpy);
   frame_cpy += 1.0f;

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


GpuWorkingSpace::GpuWorkingSpace(GpuWorkingSpaceParams params)
{
   size_t free_mem, total_mem;
   checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
   std::cout << "[Allocating working space] " << (total_mem / (1024 * 1024)) << " / " << (free_mem / (1024 * 1024)) << " Mb\n";

   error_sum.allocate(30 * params.nD);
   host_buffer.allocate(10 * params.nD);
   D.allocate(params.nD);
 
   const int n_stream = 2;

   VI_dW_dp.allocate(params.range_max * n_stream);
   error_image.allocate(params.nxny * n_stream);
   
   stream.resize(n_stream);
   for(auto& s : stream)
      cudaStreamCreate(&s);
}

GpuWorkingSpace::~GpuWorkingSpace()
{
   for(auto& s : stream)
      cudaStreamDestroy(s);
}


GpuReferenceInformation::GpuReferenceInformation(const cv::Mat& ref_, float3 offset, int nD, int range_max, bool stream_VI) :
   offset(offset), nD(nD), range_max(range_max), stream_VI(stream_VI)
{
   size_t free_mem, total_mem;
   checkCudaErrors(cudaMemGetInfo(&free_mem, &total_mem));
   std::cout << "[Allocating reference] " << (total_mem / (1024 * 1024)) << " / " << (free_mem / (1024 * 1024)) << " Mb\n";

   size_t n_px = ref_.size[0] * ref_.size[1] * ref_.size[2];

   cvref = ref_;

   VI_dW_dp_host.allocate(range_max * nD);
   reference.allocate(n_px);   
   
   checkCudaErrors(cudaMemcpy(reference, ref_.data, n_px * sizeof(float), cudaMemcpyHostToDevice));
}

WarpParams getWarpParams(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref)
{
   WarpParams p;
   p.tex_id = frame->getTextureId();
   p.size = frame->size;
   p.offset = gpu_ref->offset;
   p.reference = gpu_ref->reference;
   p.nD = gpu_ref->nD;
   p.D = w->D;
   return p;
}

void computeWarp(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref, float* warped)
{
   WarpParams p = getWarpParams(frame, w, gpu_ref);
   auto size = p.size;
   int nxny = size.x * size.y;

   dim3 dimBlock(32, 32, 1);
   dim3 dimGrid(size.x / 32, size.y / 32, size.z);

   for(int z=0; z<size.z; z++)
   {
      int stream_max = std::min((int)(w->stream.size()), size.z - z);
      for(int s=0; s<stream_max; s++)
      {
         warp<<<dimGrid, dimBlock, 0, w->stream[s]>>>(p, w->error_image + s * size.x*size.y, z+s);
         getLastCudaError("Kernel execution failed [ warp ]");      
      }
      for(int s=0; s<stream_max; s++)
         checkCudaErrors(cudaMemcpyAsync(warped + (z + s) * nxny, w->error_image + s * nxny, 
            nxny * sizeof(float), cudaMemcpyDeviceToHost, w->stream[s]));      
   }
}

/*
void computeIntensityPreservingWarp(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref, float* warped, uint16* mask)
{
   auto size = frame->size;
   dim3 dimBlock(32, 32, 1);
   dim3 dimGrid(size.x / 32, size.y / 32, size.z);

   cudaMemset(w->error_image, 0, size.x * size.y * size.z * sizeof(float));
   cudaMemset(w->mask, 0, size.x * size.y * size.z * sizeof(uint16_t));
   
   int id = frame->getTextureId();
   warpIntensityPreserving<<<dimGrid, dimBlock, 0, w->stream[0]>>>(id, frame->size, gpu_ref->offset, w->error_image, w->mask, gpu_ref->nD, w->D);
   getLastCudaError("Kernel execution failed [ warpIntensityPreserving ]");
}
*/



double computeError(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref)
{
   WarpParams p = getWarpParams(frame, w, gpu_ref);
   auto size = p.size;
   int volume = size.x * size.y * size.z;

   const int block_size = 512;
   int n_block = 1;
   reduceSumSquared<block_size><<<dim3(1,1,1), dim3(block_size,1,1), block_size * sizeof(float), w->stream[0]>>> (p, w->error_sum, volume);
   getLastCudaError("Kernel execution failed [ reduceSum ]");

   checkCudaErrors(cudaMemcpy(w->host_buffer, w->error_sum, n_block*sizeof(float), cudaMemcpyDeviceToHost));

   float error = 0;
   for(int i=0; i<n_block; i++)
      error += w->host_buffer[i];

   return error;
}

std::vector<float3> computeJacobianGpu(GpuFrame* frame, GpuWorkingSpace* w, GpuReferenceInformation* gpu_ref)
{  
   int range_max = gpu_ref->range_max;

   int n_block = 1;
   const int block_size = 512;

   int nD = gpu_ref->nD;
   WarpParams p = getWarpParams(frame, w, gpu_ref);

   size_t idx = 0;
   while(idx < nD)
   {
      int stream_max = std::min(w->stream.size(), nD - idx);
      if (gpu_ref->stream_VI)
         for(int s=0; s<stream_max; s++)
            checkCudaErrors(cudaMemcpyAsync(w->VI_dW_dp + s * range_max, gpu_ref->VI_dW_dp_host + (idx + s) * range_max, 
               range_max * sizeof(float3), cudaMemcpyHostToDevice, w->stream[s]));      

      for(int s=0; s<stream_max; s++)
      {
         auto range = gpu_ref->range[idx + s];
         float3* VI = w->VI_dW_dp + range_max * (gpu_ref->stream_VI ? s : (idx + s));
         jacobianReduceSum<block_size><<<n_block, block_size, block_size * sizeof(float3), w->stream[s]>>> 
            (((f3*) (float*) w->error_sum) + (idx + s) * n_block, p, (f3*) VI, range.end-range.begin, range);
         getLastCudaError("Kernel execution failed [ jacobianReduceSum ]");  
            
      }
      idx += w->stream.size();
   }

   for(int i=0; i<w->stream.size(); i++)
      cudaStreamSynchronize(w->stream[i]);

   float3* buffer = (float3*) (float*) w->host_buffer;
   checkCudaErrors(cudaMemcpy(buffer, w->error_sum, n_block * nD * sizeof(float3), cudaMemcpyDeviceToHost));   

   std::vector<float3> jac_out(gpu_ref->nD);
   idx = 0;
   for(int i=0; i<nD; i++)
      for(int j=0; j<n_block; j++)
         jac_out[i] += buffer[idx++];


   return jac_out;
}
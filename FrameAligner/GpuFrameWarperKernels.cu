#include "GpuFrameWarper.h"

/*
Adapted from CUDA reference
http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
*/
template <unsigned int blockSize>
__device__ void warpReduce(float *sdata, unsigned int tid) {
   if (blockSize >= 64) { sdata[tid].x += sdata[tid + 32].x; sdata[tid].y += sdata[tid + 32].y; }
   if (blockSize >= 32) { sdata[tid].x += sdata[tid + 16].x; sdata[tid].y += sdata[tid + 16].y; }
   if (blockSize >= 16) { sdata[tid].x += sdata[tid + 8].x; sdata[tid].y += sdata[tid + 8].y; }
   if (blockSize >= 8) { sdata[tid].x += sdata[tid + 4].x; sdata[tid].y += sdata[tid + 4].y; }
   if (blockSize >= 4) { sdata[tid].x += sdata[tid + 2].x; sdata[tid].y += sdata[tid + 2].y; }
   if (blockSize >= 2) { sdata[tid].x += sdata[tid + 1].x; sdata[tid].y += sdata[tid + 1].y; }
}
template <unsigned int blockSize>
__global__ void reduceSum(float *g_idata, float *g_odata, unsigned int n) {
   extern __shared__ float sdata[];
   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*(blockSize * 2) + tid;
   unsigned int gridSize = blockSize * 2 * gridDim.x;
   sdata[tid].x = 0; sdata[tid].y = 0;
   while (i < n) 
   {
      sdata[tid].x += g_idata[i].x + g_idata[i + blockSize].x; 
      sdata[tid].y += g_idata[i].x + g_idata[i + blockSize].y;
      i += gridSize;
   }
   __syncthreads();
   if (blockSize >= 512) { if (tid < 256) { sdata[tid].x += sdata[tid + 256].x; sdata[tid].y += sdata[tid + 256].y; } __syncthreads(); }
   if (blockSize >= 256) { if (tid < 128) { sdata[tid].x += sdata[tid + 128].x; sdata[tid].y += sdata[tid + 128].y; } __syncthreads(); }
   if (blockSize >= 128) { if (tid < 64) { sdata[tid].x += sdata[tid + 64].x; sdata[tid].y += sdata[tid + 64].y; } __syncthreads(); }
   if (tid < 32) warpReduce<blockSize>(sdata, tid);
   if (tid == 0) { g_odata[blockIdx.x].x = sdata[0].x; g_odata[blockIdx.x].y = sdata[0].y; };
}

__global__ void warp(float* img_tex, float* wimg, float3* D)
{
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;   
   int z = threadIdx.z + blockDim.z * blockIdx.z;
//   int idx = x + y * n_x + z * (n_y * n_z);

//   float3 p = warpPoint(D, x, y, z);

//   wimg[idx] = tex3D(img_tex, p.x, p.y, p.z);
}

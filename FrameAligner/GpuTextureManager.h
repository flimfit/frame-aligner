#pragma once

#include <cuda_runtime.h>

#include <list>
#include <mutex>
#include <condition_variable>

class GpuTextureManager
{
public:
   static GpuTextureManager* instance();

   int getTextureId();
   void returnTextureId(int t);

private:

   GpuTextureManager();

   static GpuTextureManager* gpu_texture_manager;
   std::list<int> free_textures;   
   std::mutex mutex;
   std::condition_variable cv;
};

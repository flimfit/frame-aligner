#include "FrameWarper.h"
#include "GpuFrameWarperKernels.h"
#include <list>
#include <utility>

#include <cuda_runtime.h>
#include "helper_cuda.h"


class GpuFrame
{
public:
   GpuFrame(cv::Mat frame)
   {
      // Set texture parameters
      tex.addressMode[0] = cudaAddressModeBorder;
      tex.addressMode[1] = cudaAddressModeBorder;
      tex.filterMode = cudaFilterModeLinear;
      tex.normalized = false; 

      // Allocate array and copy image data
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
      cudaExtent extent = make_cudaExtent(frame.size[Z], frame.size[Y], frame.size[X]);
      checkCudaErrors(cudaMalloc3DArray(&cu_array, &channelDesc, extent));
      checkCudaErrors(cudaMemcpyToArray(cu_array, 0, 0, frame.data, extent, cudaMemcpyHostToDevice));

      // Bind the array to the texture
      checkCudaErrors(cudaBindTextureToArray(tex, cu_array, channelDesc));
   }

   ~GpuFrame()
   {
      checkCudaErrors(cudaFree(&cu_array));
   }

   bool isSame(const cv::Mat& frame_) const { return (frame == frame_); }

protected:
   cv::Mat frame;
   texture<float, 3, cudaReadModeElementType> tex;   
   cudaArray *cu_array;
};

class GpuFrameWarper : public AbstractFrameWarper
{
public:
   double computeErrorImage(cv::Mat& wimg, cv::Mat& error_img);
   void computeJacobian(const cv::Mat& error_img, column_vector& jac);

   void warpImage(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value = 0);
   void warpImageIntensityPreserving(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D);
   void warpCoverage(cv::Mat& coverage, const std::vector<cv::Point3d>& D);

   cv::Point3d warpPoint(const std::vector<cv::Point3d>& D, int x, int y, int z, int spatial_binning = 1);

   void registerFrame(const cv::Mat& frame);
   void deregisterFrame(const cv::Mat& frame);

protected:
   void setupReferenceInformation();

   std::list<GpuFrame> frames;
};
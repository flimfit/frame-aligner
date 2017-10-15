#include "FrameWarper.h"
#include "GpuFrameWarperKernels.h"
#include <map>
#include <utility>
#include <memory>
#include <mutex>
#include "Pool.h"

#include <cuda_runtime.h>



class GpuFrameWarper : public AbstractFrameWarper
{
public:

   GpuFrameWarper();

   double getError(const cv::Mat& img, const std::vector<cv::Point3d>& D);
   void getJacobian(const cv::Mat& img, const std::vector<cv::Point3d>& D, column_vector& jac);

   void warpImage(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value = 0);
   void warpImageIntensityPreserving(const cv::Mat& img, cv::Mat& wimg, cv::Mat& coverage, const std::vector<cv::Point3d>& D);

   void registerFrame(const cv::Mat& frame);
   void deregisterFrame(const cv::Mat& frame);

   static bool hasSupportedGpu();

protected:
   void setupReferenceInformation();
   
   std::unique_ptr<GpuReferenceInformation> gpu_reference;

   std::map<void*,std::shared_ptr<GpuFrame>> frames;

   std::shared_ptr<GpuFrame> getRegisteredFrame(const cv::Mat& frame);
   std::vector<float3> D2float3(const std::vector<cv::Point3d>& D);

   bool stream_VI = false;

   int range_max;

   std::mutex mutex;

   Pool<GpuWorkingSpace,GpuWorkingSpaceParams> pool;

};
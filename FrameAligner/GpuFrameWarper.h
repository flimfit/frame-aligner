#include "FrameWarper.h"
#include "GpuFrameWarperKernels.h"
#include <map>
#include <utility>
#include <memory>

#include <cuda_runtime.h>



class GpuFrameWarper : public AbstractFrameWarper
{
public:

   GpuFrameWarper();

   double getError(const cv::Mat& img, const std::vector<cv::Point3d>& D);
   void getJacobian(const cv::Mat& img, const std::vector<cv::Point3d>& D, column_vector& jac);

   void warpImage(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value = 0);
   void warpImageIntensityPreserving(const cv::Mat& img, cv::Mat& wimg, cv::Mat& coverage, const std::vector<cv::Point3d>& D);

   int registerWorkingSpace(const std::vector<cv::Mat>& frames);
   void deregisterWorkingSpace(int space_id);

protected:
   void setupReferenceInformation();
   
   std::unique_ptr<GpuReferenceInformation> gpu_reference;

   std::map<int,std::shared_ptr<GpuWorkingSpace>> working_space;
   std::map<void*,std::shared_ptr<GpuFrame>> frames;
   std::map<void*,int> frame_space;
   int next_id = 0;

   std::shared_ptr<GpuFrame> getRegisteredFrame(const cv::Mat& frame);
   std::shared_ptr<GpuWorkingSpace> getWorkingSpace(const cv::Mat& frame);
   std::vector<float3> D2float3(const std::vector<cv::Point3d>& D);

   bool compute_jacobian_on_gpu = false;

};
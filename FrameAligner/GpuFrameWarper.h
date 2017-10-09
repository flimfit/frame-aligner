#include "FrameWarper.h"
#include "GpuFrameWarperKernels.h"
#include <list>
#include <utility>

#include <cuda_runtime.h>



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
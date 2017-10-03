#include "Cv3dUtils.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>

cv::Mat extractSlice(const cv::Mat& m, int slice)
{
   if (!((m.dims == 3) && (slice < m.size[0])))
      return cv::Mat();

   CV_Assert((m.dims == 3) && slice < m.size[0]);
   size_t offset = slice * (m.size[2] * m.size[1]) * m.elemSize();
   cv::Mat out(m.size[1], m.size[2], m.type(), m.data + offset);
   return out;
}

int area(const cv::Mat& m)
{
   int area = 1;
   for(int i=0; i<m.dims; i++)
      area *= m.size[i];
   return area;
}

void writeScaledImage(const std::string& filename, const cv::Mat& intensity)
{
   double mn, mx;
   cv::minMaxLoc(intensity, &mn, &mx);

   cv::Scalar mean, std;
   cv::meanStdDev(intensity, mean, std);
   double i_max = mean[0] + 1.96 * std[0]; // 97.5% 

   cv::Mat output;
   intensity.convertTo(output, CV_8U, 255.0 / i_max);
   
   #ifndef SUPPRESS_OPENCV_HIGHGUI
            cv::imwrite(filename, output);
   #endif
}
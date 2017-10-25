#include "Cv3dUtils.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "LinearInterpolation.h"

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
   int area = (m.dims > 0) ? 1 : 0;
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

double correlation(cv::Mat &image_1, cv::Mat &image_2, cv::Mat &mask)
{

   // convert data-type to "float"
   cv::Mat im_float_1;
   image_1.convertTo(im_float_1, CV_32F);
   cv::Mat im_float_2;
   image_2.convertTo(im_float_2, CV_32F);

   // Compute mean and standard deviation of both images
   cv::Scalar im1_Mean, im1_Std, im2_Mean, im2_Std;
   meanStdDev(im_float_1, im1_Mean, im1_Std, mask);
   meanStdDev(im_float_2, im2_Mean, im2_Std, mask);

   im_float_1 -= im1_Mean[0];
   im_float_2 -= im2_Mean[0];

   cv::multiply(im_float_1, im_float_2, im_float_1);

   // Compute covariance and correlation coefficient
   double covar = cv::mean(im_float_1, mask)[0];
   double correl = covar / (im1_Std[0] * im2_Std[0]);

   return correl;
}

void interpolatePoint3d(const std::vector<cv::Point3d>& Ds, std::vector<cv::Point3d>& D)
{
   if (Ds.empty())
      return;

   if (D.size() == Ds.size())
   {
      D = Ds;
      return;
   }

   size_t nD = D.size();
   size_t nDs = Ds.size();
   auto ud = std::vector<double>(nDs);
   auto vd_x = std::vector<double>(nDs);
   auto vd_y = std::vector<double>(nDs);
   auto vd_z = std::vector<double>(nDs);

   auto ui = std::vector<double>(nD);
   auto vi_x = std::vector<double>(nDs);
   auto vi_y = std::vector<double>(nDs);
   auto vi_z = std::vector<double>(nDs);

   for (size_t i = 0; i < nDs; i++)
   {
      ud[i] = i / (nDs - 1.0);
      vd_x[i] = Ds[i].x;
      vd_y[i] = Ds[i].y;
      vd_z[i] = Ds[i].z;
   }

   for (size_t i = 0; i < nD; i++)
      ui[i] = i / (nD - 1.0);

   pwl_interp_1d(ud, vd_x, ui, vi_x);
   pwl_interp_1d(ud, vd_y, ui, vi_y);
   pwl_interp_1d(ud, vd_z, ui, vi_z);

   for (size_t i = 0; i < nD; i++)
   {
      D[i].x = vi_x[i];
      D[i].y = vi_y[i];
      D[i].z = vi_z[i];
   }
}
#include "Cv3dUtils.h"
#include "WriteMultipageTiff.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "LinearInterpolation.h"

const int X = 2;
const int Y = 1;
const int Z = 0;


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

void writeScaledImage(const std::string& filename, const cv::Mat& intensity_)
{
   cv::Mat intensity = intensity_;
   if (intensity.dims == 2)
   {
      std::vector<int> dims = { 1, intensity.size[0], intensity.size[1] };
      intensity.reshape(intensity.channels(), 3, dims.data());
   }

   cv::Scalar mean, std;
   cv::meanStdDev(intensity, mean, std);
   double i_max = mean[0] + 1.96 * std[0]; // 97.5%

   std::vector<cv::Mat> stack;

   cv::Mat output;
   for (int z = 0; z < intensity.size[Z]; z++)
   {
      cv::Mat slice = extractSlice(intensity, z);
      slice.convertTo(output, CV_8U, 255.0 / i_max);
      stack.push_back(output);
   }

   writeMultipageTiff(filename, stack);

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

void inpaint3d(const cv::Mat& input, const cv::Mat& mask, cv::Mat& output)
{
   cv::Mat inpaint_mask, slice_exclude;
   cv::compare(mask, 0, inpaint_mask, cv::CMP_EQ);
   output = cv::Mat(3, input.size, input.type());

   int dilation_size = 2;
   cv::Mat el = cv::getStructuringElement(cv::MORPH_ELLIPSE,
      cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
      cv::Point(dilation_size, dilation_size));

   cv::Mat mask_slice_restrict;
   for (int z = 0; z < input.size[0]; z++)
   {
      // Inpaint areas within a 5px radius of another active pixel 
      cv::Mat mask_slice = extractSlice(inpaint_mask, z);
      cv::erode(mask_slice, mask_slice_restrict, el);
      cv::dilate(mask_slice_restrict, mask_slice_restrict, el);
      cv::bitwise_xor(mask_slice, mask_slice_restrict, mask_slice);

      cv::inpaint(extractSlice(input, z), mask_slice, extractSlice(output, z), 5, cv::INPAINT_NS);
   }
}

bool matsSameSizeType(const cv::Mat& a, const cv::Mat& b)
{
   if (a.type() != b.type()) return false;
   if (a.dims != b.dims) return false;
   bool same = true;
   for (int i = 0; i < a.dims; i++)
      same &= (a.size[i] == b.size[i]);
   return same;
}

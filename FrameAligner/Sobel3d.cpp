#include "Sobel3d.h"
#include "Cv3dUtils.h"

#include <array>



void applyXFilter(cv::Mat m, cv::Mat& out, std::array<float, 3> filter, std::array<float, 2> edge_filter1, std::array<float, 2> edge_filter2)
{
   for (int z = 0; z < m.size[Z]; z++)
      for (int y = 0; y < m.size[Y]; y++)
         for (int x = 1; x < (m.size[X] - 1); x++)
            out.at<float>(z, y, x) = filter[0] * m.at<float>(z, y, x - 1) + filter[1] * m.at<float>(z, y, x) + filter[2] * m.at<float>(z, y, x + 1);

   for (int z = 0; z < m.size[Z]; z++)
      for (int y = 0; y < m.size[Y]; y++)
      {
         out.at<float>(z, y, 0) = edge_filter1[0] * m.at<float>(z, y, 0) + edge_filter1[1] * m.at<float>(z, y, 1);
         out.at<float>(z, y, m.size[X] - 1) = edge_filter2[0] * m.at<float>(z, y, m.size[X] - 1) + edge_filter2[1] * m.at<float>(z, y, m.size[X] - 2);
      }
}

void applyYFilter(cv::Mat m, cv::Mat& out, std::array<float, 3> filter, std::array<float, 2> edge_filter1, std::array<float, 2> edge_filter2)
{
   for (int z = 0; z < m.size[Z]; z++)
      for (int y = 1; y < (m.size[Y] - 1); y++)
         for (int x = 0; x < m.size[X]; x++)
            out.at<float>(z, y, x) = filter[0] * m.at<float>(z, y - 1, x) + filter[1] * m.at<float>(z, y, x) + filter[2] * m.at<float>(z, y + 1, x);

   for (int z = 0; z < m.size[Z]; z++)
      for (int x = 0; x < m.size[X]; x++)
      {
         out.at<float>(z, 0, x) = edge_filter1[0] * m.at<float>(z, 0, x) + edge_filter1[1] * m.at<float>(z, 1, x);
         out.at<float>(z, m.size[Y] - 1, x) = edge_filter2[0] * m.at<float>(z, m.size[Y] - 1, x) + edge_filter2[1] * m.at<float>(z, m.size[Y] - 2, x);
      }
}

void applyZFilter(cv::Mat m, cv::Mat& out, std::array<float, 3> filter, std::array<float, 2> edge_filter1, std::array<float, 2> edge_filter2)
{
   if (m.size[Z] == 1)
   {
      m.copyTo(out);
      return;
   }

   for (int z = 1; z < (m.size[Z] - 1); z++)
      for (int y = 0; y < m.size[Y]; y++)
         for (int x = 0; x < m.size[X]; x++)
            out.at<float>(z, y, x) = filter[0] * m.at<float>(z - 1, y, x) + filter[1] * m.at<float>(z, y, x) + filter[2] * m.at<float>(z + 1, y, x);


   for (int y = 0; y < m.size[Y]; y++)
      for (int x = 0; x < m.size[X]; x++)
      {
         out.at<float>(0, y, x) = edge_filter1[0] * m.at<float>(0, y, x) + edge_filter1[1] * m.at<float>(1, y, x);
         out.at<float>(m.size[Z] - 1, y, 0) = edge_filter2[0] * m.at<float>(m.size[Z] - 1, y, 0) + edge_filter2[1] * m.at<float>(m.size[Z] - 2, y, 0);
      }
}

void sobel3d(cv::Mat m, cv::Mat& gx, cv::Mat& gy, cv::Mat& gz)
{
   cv::Mat a(m.dims, m.size, m.type()), b(m.dims, m.size, m.type()), c(m.dims, m.size, m.type());

   std::array<float, 3> a1 = { 1.0/4.0, 2.0/4.0, 1.0/4.0 };
   std::array<float, 2> a2 = { 2.0/3.0, 2.0/3.0 };
   std::array<float, 2> a3 = { 2.0 / 3.0, 2.0 / 3.0 };

   std::array<float, 3> s1 = { -0.5, 0, 0.5 };
   std::array<float, 2> s2 = { -0.5, 0.5 };
   std::array<float, 2> s3 = { 0.5, -0.5 };


   applyXFilter(m, a, a1, a2, a3);

   applyYFilter(a, b, a1, a2, a3);
   applyZFilter(a, c, a1, a2, a3);

   applyZFilter(b, gz, s1, s2, s3);
   applyYFilter(c, gy, s1, s2, s3);

   applyYFilter(m, a, a1, a2, a3);
   applyZFilter(a, b, a1, a2, a3);

   applyXFilter(b, gx, s1, s2, s3);
}
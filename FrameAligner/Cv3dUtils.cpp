#include "Cv3dUtils.h"

cv::Mat extractSlice(const cv::Mat& m, int slice)
{
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
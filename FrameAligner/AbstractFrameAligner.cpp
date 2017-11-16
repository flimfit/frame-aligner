#include "RigidFrameAligner.h"
#include "FrameWarpAligner.h"
#include "Cv3dUtils.h"

std::string realignmentTypeString(RealignmentType t)
{
   switch (t)
   {
   case RealignmentType::None:
      return "None";
   case RealignmentType::Translation:
      return "Translation";
   //case RealignmentType::RigidBody:
   //   return "RigidBody";
   case RealignmentType::Warp:
      return "Warp";
   default:
      return "";
   }
}


AbstractFrameAligner* AbstractFrameAligner::createFrameAligner(RealignmentParameters params)
{
   switch (params.type)
   {
   case RealignmentType::None:
      return nullptr;
   //case RealignmentType::RigidBody:
   //   throw std::runtime_error("Rigid body is not currently supported");
   default:
      return new FrameWarpAligner(params);
      // Let FrameWarpAligner handle translation as well, not supporting rigid body currently
   }
}

cv::Mat downsample(const cv::Mat& im1, int factor)
{
   int nz = im1.size[Z];
   int w = im1.size[X];
   int h = im1.size[Y];

   int nh = ceil(h / (double)factor);
   int nw = ceil(w / (double)factor);

   std::vector<int> new_size = {nz, nh, nw};

   cv::Mat im2(new_size, im1.type(), cv::Scalar(0));

   for (int z = 0; z < nz; z++)
      for (int y = 0; y < h; y++)
         for (int x = 0; x < w; x++)
            im2.at<float>(z, y / factor, x / factor) += im1.at<float>(z, y, x);

   return im2;
}

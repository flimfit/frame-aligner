#include "OptimisationModel.h"


OptimisationModel::OptimisationModel(FrameWarpAligner* aligner, const cv::Mat& frame, const cv::Mat& raw_frame) :
aligner(aligner),
raw_frame(raw_frame),
frame(frame),
realign_params(aligner->realign_params)
{
   error_buffer = std::make_unique<std::list<std::pair<column_vector, cv::Mat>>>();
}

void D2col(const std::vector<cv::Point3d> &D, column_vector& col, int n_dim)
{
   int nD = D.size();
   col.set_size(nD * n_dim);

   int idx = 0;
   for (int i = 0; i < nD; i++)
   {
      col(idx++) = D[i].x;
      col(idx++) = D[i].y;
      if (n_dim == 3)
         col(idx++) = D[i].z;
   }
}

void col2D(const column_vector& col, std::vector<cv::Point3d> &D, int n_dim)
{
   int nD = col.size() / n_dim;
   D.resize(nD);

   int idx = 0;
   for (int i = 0; i < nD; i++)
   {
      D[i].x = col(idx++);
      D[i].y = col(idx++);
      if (n_dim == 3) 
         D[i].z = col(idx++);
   }
}


double OptimisationModel::operator() (const column_vector& x) const
{
   std::vector<cv::Point3d> D;
   cv::Mat warped_image, error_image;

   // Get displacement matrix from warp parameters
   col2D(x, D, aligner->n_dim);

   aligner->warpImage(frame, warped_image, D, -1);
   double rms_error = aligner->computeErrorImage(warped_image, error_image);

   auto p = std::make_pair(x, error_image);
   error_buffer->push_front(p);

   if (error_buffer->size() > 2)
      error_buffer->pop_back();

   return rms_error;
}

void OptimisationModel::get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const
{
   std::vector<cv::Point3d> D;
   cv::Mat warped_image, error_image;

   col2D(x, D, aligner->n_dim);
   auto it = std::find_if(error_buffer->begin(), error_buffer->end(), [x](auto& it) -> bool { return it.first == x; });

   if (it != error_buffer->end())
   {
      error_image = it->second;
   }
   else
   {
      aligner->warpImage(frame, warped_image, D, -1);
      aligner->computeErrorImage(warped_image, error_image);
   }

   aligner->computeJacobian(error_image, der);
   hess = aligner->H;
}

cv::Mat OptimisationModel::getMask(const column_vector& x)
{
   std::vector<cv::Point3d> D;
   col2D(x, D, aligner->n_dim);

   cv::Mat mask;
   aligner->warpCoverage(mask, D);
   return mask;
}

cv::Mat OptimisationModel::getWarpedRawImage(const column_vector& x)
{
   std::vector<cv::Point3d> D;
   col2D(x, D, aligner->n_dim);
   cv::Mat warped_image;
   aligner->warpImage(raw_frame, warped_image, D);
   return warped_image;
}

cv::Mat OptimisationModel::getWarpedImage(const column_vector& x)
{
   std::vector<cv::Point3d> D;
   col2D(x, D, aligner->n_dim);
   cv::Mat warped_image;
   aligner->warpImage(frame, warped_image, D);
   return warped_image;
}

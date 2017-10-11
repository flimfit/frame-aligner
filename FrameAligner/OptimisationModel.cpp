#include "OptimisationModel.h"


OptimisationModel::OptimisationModel(std::shared_ptr<AbstractFrameWarper> warper, const cv::Mat& frame, const cv::Mat& raw_frame) :
warper(warper),
raw_frame(raw_frame),
frame(frame)
{
   warper->registerFrame(frame);
   warper->registerFrame(raw_frame);
}

OptimisationModel::~OptimisationModel()
{
   warper->deregisterFrame(frame);
   warper->deregisterFrame(raw_frame);
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
   // Get displacement matrix from warp parameters
   col2D(x, D, warper->n_dim);

   double rms_error = warper->getError(frame, D);




   return rms_error;
}

void OptimisationModel::get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const
{
   std::vector<cv::Point3d> D;
   col2D(x, D, warper->n_dim);

   warper->getJacobian(frame, D, der);
   hess = warper->H;
}


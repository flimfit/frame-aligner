#pragma once
#include "FrameWarper.h"
#include <list>
#include <utility>

class OptimisationModel
{
   /*!
   This object is a "function model" which can be used with the
   find_min_trust_region() routine.
   !*/

public:
   typedef ::column_vector column_vector;
   typedef matrix<double> general_matrix;

   OptimisationModel(std::shared_ptr<AbstractFrameWarper> warper, const cv::Mat& frame, const cv::Mat& raw_frame);

   double operator() (const column_vector& x) const;
   void get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const;

   cv::Mat getMask(const column_vector& x);
   cv::Mat getWarpedRawImage(const column_vector& x);
   cv::Mat getWarpedImage(const column_vector& x);

protected:

   std::shared_ptr<AbstractFrameWarper> warper;
   cv::Mat raw_frame;
   cv::Mat frame;
};

void D2col(const std::vector<cv::Point3d> &D, column_vector& col, int n_dim);
void col2D(const column_vector& col, std::vector<cv::Point3d> &D, int n_dim);
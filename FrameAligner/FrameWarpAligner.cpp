#include "FrameWarpAligner.h"
#include "LinearInterpolation.h"
#include "Cv3dUtils.h"

#include <functional>
#include <fstream>

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

   int nD = D.size();
   int nDs = Ds.size();
   auto ud = std::vector<double>(nDs);
   auto vd_x = std::vector<double>(nDs);
   auto vd_y = std::vector<double>(nDs);
   auto vd_z = std::vector<double>(nDs);

   auto ui = std::vector<double>(nD);
   auto vi_x = std::vector<double>(nDs);
   auto vi_y = std::vector<double>(nDs);
   auto vi_z = std::vector<double>(nDs);

   for (int i = 0; i < nDs; i++)
   {
      ud[i] = i / (nDs - 1.0);
      vd_x[i] = Ds[i].x;
      vd_y[i] = Ds[i].y;
      vd_z[i] = Ds[i].z;
   }

   for (int i = 0; i < nD; i++)
      ui[i] = i / (nD - 1.0);

   pwl_interp_1d(ud, vd_x, ui, vi_x);
   pwl_interp_1d(ud, vd_y, ui, vi_y);
   pwl_interp_1d(ud, vd_z, ui, vi_z);

   for (int i = 0; i < nD; i++)
   {
      D[i].x = vi_x[i];
      D[i].y = vi_y[i];
      D[i].z = vi_z[i];
   }
}

FrameWarpAligner::FrameWarpAligner(RealignmentParameters params)
{
   realign_params = params;
}

cv::Mat FrameWarpAligner::reshapeForOutput(cv::Mat& m)
{
   if ((dims[Z] == 1) && output2d)
      return m.reshape(0, 2, &dims[Y]);
   else
      return m;
}

cv::Mat FrameWarpAligner::reshapeForProcessing(cv::Mat& m)
{
   return m.reshape(0, 3, dims.data());
}


void FrameWarpAligner::smoothStack(const cv::Mat& in, cv::Mat& out)
{

   if (realign_params.smoothing > 0.0)
   {
      out = cv::Mat(dims, CV_32F);
      cv::Mat buf;
      for (int i = 0; i < dims[Z]; i++)
      {
         extractSlice(in, i).convertTo(buf, CV_32F);
         cv::GaussianBlur(buf, extractSlice(out, i), cv::Size(0, 0), realign_params.smoothing, 1);
      }
   }
   else
   {
      in.convertTo(out, CV_32F);
   }
   
}

bool FrameWarpAligner::frameReady(int frame)
{
   if (frame >= n_frames) return false;
   return (results[frame].done);
}

void FrameWarpAligner::setNumberOfFrames(int n_frames_)
{
   n_frames = n_frames_;

   Dstore.clear();
   results.clear();
   
   Dstore.resize(n_frames, std::vector<cv::Point3d>(nD));
   results.resize(n_frames);
}


void FrameWarpAligner::setReference(int frame_t, const cv::Mat& reference_)
{
   output2d = (reference_.dims == 2);

   n_x_binned = image_params.n_x / realign_params.spatial_binning;
   n_y_binned = image_params.n_y / realign_params.spatial_binning;

   dims = {image_params.n_z, n_y_binned, n_x_binned};
   n_dim = (image_params.n_z > 1) ? 3 : 2;

   phase_correlator = std::unique_ptr<VolumePhaseCorrelator>(new VolumePhaseCorrelator(dims[Z], dims[Y] / 4, dims[X] / 4));   

   reference_.copyTo(reference);
   reference.convertTo(reference, CV_32F);

   reference = reshapeForProcessing(reference);   
   smoothStack(reference, smoothed_reference);

   cv::Mat f = downsample(reference, phase_downsampling);
   phase_correlator->setReference((float*) f.data);

   nD = realign_params.n_resampling_points;

   precomputeInterp();
   computeSteepestDecentImages(smoothed_reference);
   computeHessian();

   Dlast = cv::Point3d(0, 0, 0);
}

void FrameWarpAligner::reprocess()
{
   sum_2 /= (double) results.size();
   setReference(0, sum_2);

//   for (auto iter : results)
//      addFrame(iter.first, iter.second.frame);
}

void D2col(const std::vector<cv::Point3d> &D, column_vector& col, int n_dim)
{
   int nD = D.size();
   col.set_size(nD * n_dim);

   for (int i = 0; i < nD; i++)
   {
      col(i) = D[i].x;
      col(i + nD) = D[i].y;
      if (n_dim == 3)
            col(i + 2*nD) = D[i].z;
   }
}

void col2D(const column_vector& col, std::vector<cv::Point3d> &D, int n_dim)
{
   int nD = col.size() / n_dim;
   D.resize(nD);

   for (int i = 0; i < nD; i++)
   {
      D[i].x = col(i);
      D[i].y = col(i + nD);
      if (n_dim == 3) D[i].z = col(i + nD);
   }
}


RealignmentResult FrameWarpAligner::addFrame(int frame_t, const cv::Mat& raw_frame_)
{
   cv::Mat raw_frame, frame;
   raw_frame_.copyTo(raw_frame);
   raw_frame.convertTo(raw_frame, CV_32F);

   raw_frame = reshapeForProcessing(raw_frame);
   smoothStack(raw_frame, frame);

   auto model = OptimisationModel(this, frame, raw_frame);

   std::vector<column_vector> starting_point(2, column_vector(nD * n_dim));

   // zero starting point
   std::fill(starting_point[0].begin(), starting_point[0].end(), 0);

   std::vector<cv::Point3d> D(nD, cv::Point3d(0,0,0));
   // last starting point
   //if (frame_t < n_frames)
   //   interpolatePoint3d(Dstore[frame_t], D);
   //D2col(D, starting_point[1], n_dim);

   // rigid starting point
   cv::Mat ff = downsample(raw_frame, phase_downsampling);
   cv::Point3d rigid_shift = phase_correlator->computeShift((float*) ff.data);
   rigid_shift.x *= phase_downsampling;
   rigid_shift.y *= phase_downsampling;
   std::vector<cv::Point3d> D_rigid(nD, rigid_shift);
   D2col(D_rigid, starting_point[1], n_dim);
   
   std::vector<double> point_fcn(starting_point.size());
   double best = std::numeric_limits<double>::max();
   int best_start = 0;
   for (int i = 0; i < starting_point.size(); i++)
   {
      double new_value = model(starting_point[i]);
      point_fcn[i] = new_value;
      if (new_value < best)
      {
         best_start = i;
         best = new_value;
      }
   }

   best_start = 0;

   column_vector x = starting_point[best_start];

   auto f_der = [&](const column_vector& x) -> column_vector 
   {
      column_vector der;
      matrix<double> hess;
      model.get_derivative_and_hessian(x, der, hess);
      return der;
   };

   auto f = [&](const column_vector& x) -> double
   {
      return model(x);
   };


   /*
   auto d1 = derivative(f, 4)(x);
   column_vector d2 = f_der(x);
   double l = length(d1 - d2);

   std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
      << l << std::endl;
   */   

   
   try
   {
	   
      find_min_trust_region(dlib::objective_delta_stop_strategy(1e-5),
         model,
         x,
         40 // initial trust region radius
      );
   }
   catch (dlib::error e)
   {
      std::cout << e.info;
   }
   
   col2D(x, D, n_dim);
   Dstore[frame_t] = D;
   Dlast = *(D.end() - 2);

   std::cout << "*";

   cv::Mat warped_smoothed = model.getWarpedImage(x);
   cv::Mat warped = model.getWarpedRawImage(x);
   cv::Mat mask = model.getMask(x);

   cv::Mat m;
   cv::compare(mask, 0, m, cv::CMP_GT);
   

   RealignmentResult r;
   r.frame = reshapeForOutput(raw_frame);
   r.realigned = reshapeForOutput(warped);
   r.mask = reshapeForOutput(mask);
   r.correlation = correlation(warped_smoothed, smoothed_reference, m);
   r.unaligned_correlation = correlation(frame, smoothed_reference, m);
   r.coverage = ((double)cv::countNonZero(mask)) / (dims[X] * dims[Y] * dims[Z]);
   
   cv::Mat intensity_preserving(frame.dims, frame.size, CV_16U, cv::Scalar(0));
   if (r.correlation >= realign_params.correlation_threshold && r.coverage >= realign_params.coverage_threshold)
      warpImageIntensityPreserving(raw_frame, intensity_preserving, D);
   r.realigned_preserving = reshapeForOutput(intensity_preserving);
   r.done = true;
   
   results[frame_t] = r;

   return r;


}

OptimisationModel::OptimisationModel(FrameWarpAligner* aligner, const cv::Mat& frame, const cv::Mat& raw_frame) :
   aligner(aligner),
   raw_frame(raw_frame),
   frame(frame),
   realign_params(aligner->realign_params)
{
}


double OptimisationModel::operator() (const column_vector& x) const
{
   std::vector<cv::Point3d> D;
   cv::Mat warped_image, error_image;
   
   // Get displacement matrix from warp parameters
   col2D(x, D, aligner->n_dim);

   aligner->warpImage(frame, warped_image, D, -1);
   double rms_error = aligner->computeErrorImage(warped_image, error_image);

   std::cout << "E: " << rms_error << "\n";
   return rms_error;
}

void OptimisationModel::get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const
{
   std::vector<cv::Point3d> D;
   cv::Mat warped_image, error_image;

   // Get displacement matrix from warp parameters
   col2D(x, D, aligner->n_dim);

   aligner->warpImage(frame, warped_image, D, -1);
   double rms_error = aligner->computeErrorImage(warped_image, error_image);

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

double FrameWarpAligner::computeErrorImage(cv::Mat& wimg, cv::Mat& error_img)
{
   int n_px = dims[X] * dims[Y] * dims[Z];

   error_img = wimg - smoothed_reference;
   double n_include = 0;

   double ms_error = 0;
   for(int z=0; z<dims[Z]; z++)
      for (int y = 0; y<dims[Y]; y++)
         for (int x = 0; x < dims[X]; x++)
         {
            if (wimg.at<float>(z, y, x) >= 0)
            {
               ms_error += error_img.at<float>(z, y, x) * error_img.at<float>(z, y, x);
               n_include++;
            }
            else
            {
               error_img.at<float>(z, y, x) = 0;
               wimg.at<float>(z, y, x) = 0;
            }
         }

   return ms_error;
}

void FrameWarpAligner::shiftPixel(int frame_t, double& x, double& y, double& z)
{
   if (!realign_params.use_realignment())
      return;

   assert(Dstore.size() > frame_t);

   auto& D = Dstore[frame_t];
   cv::Point3d loc = warpPoint(D, x, y, z);

   x -= loc.x;
   y -= loc.y;
   z -= loc.z;
}



/*
Jacobian structure:

0 1 2 3 4 5
|/|/|/|/|/|
-----------
0:|\|
1:|/|\|
2:  |/|\|
3:    |/|\|
4:      |/|\|
5:        |/|
*/


void FrameWarpAligner::precomputeInterp()
{
   double pixel_duration = image_params.pixel_duration * realign_params.spatial_binning;
   double frame_duration = image_params.frame_duration;
   double interline_duration = image_params.interline_duration * realign_params.spatial_binning;
   double stack_duration = dims[Z] * frame_duration;

   D_range.resize(nD);

   Di = cv::Mat(dims, CV_16U);
   Df = cv::Mat(dims, CV_32F);
   
   double Di_xy;
   int i;
   int last_i = -1;

   for (int z = 0; z < dims[Z]; z++)
   {
      for (int y = 0; y < dims[Y]; y++)
      {
         for (int x = 0; x < dims[X]; x++)
         {
            double t = z * frame_duration + y * interline_duration + x * pixel_duration;
            double f = modf(t / stack_duration * (nD - 1), &Di_xy);
            i = (int)Di_xy;

            int x_true = x;
            if (image_params.bidirectional && ((y % 2) == 1))
               x_true = dims[X] - x - 1;

            Di.at<uint16_t>(z, y, x_true) = i;
            Df.at<float>(z, y, x_true) = f;

            if (i > last_i)
            {
               D_range[i].begin = (z*dims[Y] + y)*dims[X] + x_true;
               if (last_i >= 0)
                  D_range[last_i].end = D_range[i].begin - 1;
               last_i = i;
            }
         }
      }
   }

   D_range[i].end = dims[X]*dims[Y]*dims[Z] - 1;

   int max_interval = 0;
   for (int i = 0; i < nD; i++)
   {
      int interval = D_range[i].interval();
      if (interval > max_interval)
         max_interval = D_range[i].interval();

   }

   int max_VI_dW_dp = dims[X] * dims[Y] * dims[Z];
   max_VI_dW_dp = max_interval * 2 + 1;

   VI_dW_dp_x.clear();
   VI_dW_dp_y.clear();
   VI_dW_dp_z.clear();

   for (int i = 0; i < nD; i++)
   {
      int i0 = D_range[std::max(0, i - 1)].begin;
      VI_dW_dp_x.push_back(OffsetVector<float>(max_VI_dW_dp, i0));
      VI_dW_dp_y.push_back(OffsetVector<float>(max_VI_dW_dp, i0));
      VI_dW_dp_z.push_back(OffsetVector<float>(max_VI_dW_dp, i0));
   }
}


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
         out.at<float>(z, 0, x) = edge_filter1[0] * m.at<float>(z, 0, x) + edge_filter1[1] * m.at<float>(z, 0, x);
            out.at<float>(z, m.size[Y] - 1, 0) = edge_filter2[0] * m.at<float>(z, m.size[Y] - 1, 0) + edge_filter2[1] * m.at<float>(z, m.size[Y] - 2, 0);
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


void FrameWarpAligner::computeSteepestDecentImages(const cv::Mat& frame)
{
   // Evaluate gradient of reference

   cv::Mat nabla_Tx(dims, CV_32F);
   cv::Mat nabla_Ty(dims, CV_32F);
   cv::Mat nabla_Tz(dims, CV_32F);

   cv::Mat nabla_Tx1(dims, CV_32F);
   cv::Mat nabla_Ty1(dims, CV_32F);
   cv::Mat nabla_Tz1(dims, CV_32F);


   sobel3d(frame, nabla_Tx, nabla_Ty, nabla_Tz);

   /*
   auto padded_dims = dims;
   padded_dims[Z]++;
   cv::Mat d0(padded_dims, frame.type(), cv::Scalar(0));
   std::copy_n(frame.data, dims[Z] * dims[Y] * dims[X] * frame.elemSize1(), d0.data);

   cv::Mat dx(dims, frame.type(), d0.data + frame.elemSize1());
   cv::Mat dy(dims, frame.type(), d0.data + dims[X] * frame.elemSize1());
   cv::Mat dz(dims, frame.type(), d0.data + dims[Y] * dims[X] * frame.elemSize1());

   nabla_Tx1 = (dx - frame);
   nabla_Ty1 = (dy - frame);
   nabla_Tz1 = (dz - frame);

   // Replace invalid entries at end of dimensions
   for (int z = 0; z < dims[Z]; z++)
   {
      for (int y = 0; y < dims[Y]; y++)
         nabla_Tx1.at<float>(z, y, dims[X] - 1) = nabla_Tx1.at<float>(z, y, dims[X] - 2);

      for (int x = 0; x < dims[X]; x++)
         nabla_Ty1.at<float>(z, dims[Y] - 1, x) = nabla_Ty1.at<float>(z, dims[Y] - 2, x);
   }

   if (dims[Z] > 1)
      extractSlice(nabla_Tz1, dims[Z] - 2).copyTo(extractSlice(nabla_Tz1, dims[Z] - 1));
   */

   //================================================
   /*
   for (int i = 0; i < dims[Z]; i++)
   {
      cv::Mat frame_i = extractSlice(frame, i);

      cv::Mat nabla_Tx_i = extractSlice(nabla_Tx, i);
      cv::Mat nabla_Ty_i = extractSlice(nabla_Ty, i);
      cv::Mat nabla_Tz_i = extractSlice(nabla_Tz, i);

      cv::Scharr(frame_i, nabla_Tx_i, CV_32F, 1, 0, 1.0 / 32.0);
      cv::Scharr(frame_i, nabla_Ty_i, CV_32F, 0, 1, 1.0 / 32.0);

      if (i < (dims[Z] - 1))
      {
         cv::Mat frame_i1 = extractSlice(frame, i + 1);
         cv::Mat diff = frame_i - frame_i1;
         diff.convertTo(nabla_Tz_i, CV_32F);
      }
      else if (dims[Z] > 1)
      {
         cv::Mat nabla_Tz_last = extractSlice(nabla_Tz, i - 1);
         nabla_Tz_last.copyTo(nabla_Tz_i);
      }
   }
   */

   //====================


   float* nabla_Txd = (float*)nabla_Tx.data;
   float* nabla_Tyd = (float*)nabla_Ty.data;
   float* nabla_Tzd = (float*)nabla_Tz.data;
   float* Dfd = (float*)Df.data;

   //#pragma omp parallel for
   for (int i = 1; i < nD; i++)
   {
      int p0 = D_range[i - 1].begin;
      int p1 = D_range[i - 1].end;
      for (int p = p0; p < p1; p++)
      {
         double jac = Dfd[p];
         VI_dW_dp_x[i][p] = nabla_Txd[p] * jac;
         VI_dW_dp_y[i][p] = nabla_Tyd[p] * jac;

         if (n_dim == 3)
            VI_dW_dp_z[i][p] = nabla_Tzd[p] * jac;
      }
   }
   
   //#pragma omp parallel for
   for(int i = 0; i < (nD - 1); i++)
   {
      int p0 = D_range[i].begin;
      int p1 = D_range[i].end;
      for (int p = p0; p < p1; p++)
      {
         double jac = 1 - Dfd[p];
         VI_dW_dp_x[i][p] = nabla_Txd[p] * jac;
         VI_dW_dp_y[i][p] = nabla_Tyd[p] * jac;

         if (n_dim == 3)
            VI_dW_dp_z[i][p] = nabla_Tzd[p] * jac;
      }
   }
}

double FrameWarpAligner::computeHessianEntry(int pi, int pj)
{
   int i = pi % nD;
   int j = pj % nD;

   if (j > i) return computeHessianEntry(pj, pi);
   if ((i - j) > 1) return 0;

   auto getV = [this](int i) -> const OffsetVector<float>& {
      if (i < nD)
         return VI_dW_dp_x[i];
      else if (i < 2*nD)
         return VI_dW_dp_y[i - nD];
      else
         return VI_dW_dp_z[i - 2*nD];
      };

   auto v1 = getV(pi);
   auto v2 = getV(pj);
  
   int p0 = std::max(v1.first(), v2.first());
   int p1 = std::min(v1.last(), v2.last());

   double h = 0;
   for (int p = p0; p < p1; p++)
      h += v1.at(p) * v2.at(p);
      
   return h;
}

void FrameWarpAligner::computeHessian()
{   
   H.set_size(n_dim * nD, n_dim * nD);
   //std::fill(H.begin(), H.end(), 0);

   // Diagonal elements
   for (int i = 0; i < nD * n_dim; i++)
      for (int j = 0; j < nD * n_dim; j++)
      {
         double h = computeHessianEntry(i, j);
         H(i, j) = h;
         H(j, i) = h;
      }

   /*
   // Off diagonal elements
   for (int pi = 1; pi < nD * n_dim; pi++)
   {
      double h = computeHessianEntry(pi, pi - 1);
      H(pi, pi - 1) = h;
      H(pi - 1, pi) = h;
   }
   
   // Interactions between x,y,z
   for (int i = 0; i < nD; i++)
   {
      for (int j = std::max(0, i - 1); j < std::min(nD, i + 1); j++)
      {
         for (int d=1; d < n_dim; d++)
         {
            double h = computeHessianEntry(i, j + d * nD);
            H(i, j + d * nD) += h;
            H(j + d * nD, i) += h;
         }
      }
   }
   */

}


void FrameWarpAligner::computeJacobian(const cv::Mat& error_img, column_vector& jac)
{
   jac.set_size(nD * n_dim);
   std::fill(jac.begin(), jac.end(), 0);

   // we are ignoring stride here, ok as set up
   float* err_ptr = reinterpret_cast<float*>(error_img.data);

   for (int i = 1; i < nD; i++)
   {
      int p0 = D_range[i - 1].begin;
      int p1 = D_range[i - 1].end;
      for (int p = p0; p < p1; p++)
      {
         jac(i) += VI_dW_dp_x[i][p] * err_ptr[p]; // x 
         jac(i+nD) += VI_dW_dp_y[i][p] * err_ptr[p]; // y
         if (n_dim == 3)
            jac(i+2*nD) += VI_dW_dp_z[i][p] * err_ptr[p]; // z        
      }
   }
   for (int i = 0; i < (nD - 1); i++)
   {
      int p0 = D_range[i].begin;
      int p1 = D_range[i].end;
      for (int p = p0; p < p1; p++)
      {
         jac(i) += VI_dW_dp_x[i][p] * err_ptr[p]; // x
         jac(i+nD) += VI_dW_dp_y[i][p] * err_ptr[p]; // y
         if (n_dim == 3)
            jac(i+2*nD) += VI_dW_dp_z[i][p] * err_ptr[p]; // z
         
      }
   }
}


void FrameWarpAligner::warpImage(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value)
{
   wimg = cv::Mat(dims, CV_32F, cv::Scalar(invalid_value));
   
   for (int z = 0; z < dims[Z]; z++)
      for (int y = 0; y < dims[Y]; y++)
         for (int x = 0; x < dims[X]; x++)
         {
            cv::Point3d loc = warpPoint(D, x, y, z, realign_params.spatial_binning);
            loc += cv::Point3d(x,y,z);

            // Clamp values slightly outside range
            if ((loc.x < 0) && (loc.x > -1)) loc.x = 0;
            if ((loc.y < 0) && (loc.y > -1)) loc.y = 0;
            if ((loc.z < 0) && (loc.z > -1)) loc.z = 0;

            if ((loc.x > (dims[X] - 1)) && (loc.x < dims[X])) loc.x = dims[X]-1;
            if ((loc.y > (dims[Y] - 1)) && (loc.y < dims[Y])) loc.y = dims[Y]-1;
            if ((loc.z > (dims[Z] - 1)) && (loc.z < dims[Z])) loc.z = dims[Z]-1;

            cv::Point3i loc0(floor(loc.x), floor(loc.y), floor(loc.z));
            cv::Point3d locf(loc.x - loc0.x, loc.y - loc0.y, loc.z - loc0.z);

            cv::Point3i loc1 = loc0;
            if (loc1.x < (dims[X]-1)) loc1.x++;
            if (loc1.y < (dims[Y]-1)) loc1.y++;
            if (loc1.z < (dims[Z]-1)) loc1.z++;
            
            if (isValidPoint(loc0))
            {
               wimg.at<float>(z, y, x) =
                  (1 - locf.z) * (
                     img.at<float>(loc0.z, loc0.y, loc0.x) * (1 - locf.y) * (1 - locf.x) + 
                     img.at<float>(loc0.z, loc1.y, loc0.x) * (    locf.y) * (1 - locf.x) + 
                     img.at<float>(loc0.z, loc0.y, loc1.x) * (1 - locf.y) * (    locf.x) + 
                     img.at<float>(loc0.z, loc1.y, loc1.x) * (    locf.y) * (    locf.x)
                  );
               if (n_dim == 3)
                  wimg.at<float>(z, y, x) +=
                     locf.z * (
                        img.at<float>(loc1.z, loc0.y, loc0.x) * (1 - locf.y) * (1 - locf.x) + 
                        img.at<float>(loc1.z, loc1.y, loc0.x) * (    locf.y) * (1 - locf.x) + 
                        img.at<float>(loc1.z, loc0.y, loc1.x) * (1 - locf.y) * (    locf.x) + 
                        img.at<float>(loc1.z, loc1.y, loc1.x) * (    locf.y) * (    locf.x)
                     );
            }
               
         }
}

void FrameWarpAligner::warpImageIntensityPreserving(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D)
{
   wimg = cv::Mat(dims, CV_16U, cv::Scalar(0));

   for (int z = 0; z < dims[Z]; z++)
      for (int y = 0; y < dims[Y]; y++)
         for (int x = 0; x < dims[X]; x++)
         {
            cv::Point3d loc = warpPoint(D, x, y, z, realign_params.spatial_binning);

            cv::Vec<int,3> locr = { (int)round(z - loc.z),
                                    (int)round(y - loc.y),
                                    (int)round(x - loc.x) };              
               
            if (isValidPoint(locr))
               wimg.at<uint16_t>(locr) += img.at<float>(z, y, x);
         }
}


void FrameWarpAligner::warpCoverage(cv::Mat& coverage, const std::vector<cv::Point3d>& D)
{
   coverage = cv::Mat(dims, CV_16U, cv::Scalar(0));

   for(int z = 0; z < dims[Z]; z++)
   for (int y = 0; y < dims[Y]; y++)
      for (int x = 0; x < dims[X]; x++)
      {
         cv::Point3d loc = warpPoint(D, x, y, z, realign_params.spatial_binning);

         cv::Vec<int,3> locr = { (int)round(z - loc.z),
                                 (int)round(y - loc.y),
                                 (int)round(x - loc.x) };
                  
         if (isValidPoint(locr))
            coverage.at<uint16_t>(locr)++;
      }
}

cv::Point3d FrameWarpAligner::warpPoint(const std::vector<cv::Point3d>& D, int x, int y, int z, int spatial_binning)
{
   double factor = ((double)realign_params.spatial_binning) / spatial_binning;
   int xs = (int) (x / factor);
   int ys = (int) (y / factor);

   if (   (xs < 0) || (xs >= dims[X]) 
       || (ys < 0) || (ys >= dims[Y])
       || (z  < 0) || (z  >= dims[Z]))
      return cv::Point3d(0,0,0);

   float* Df_d = reinterpret_cast<float*>(Df.data);
   uint16_t* Di_d = reinterpret_cast<uint16_t*>(Di.data);
   int loc = (z * dims[Y] + ys) * dims[X] + xs;

   double f = Df_d[loc];
   int i = Di_d[loc];

   if (i >= (nD-1))
      int a = 1;

   cv::Point3d p = f * D[i + 1] + (1 - f) * D[i];
   p *= factor;

   return p;
}

void FrameWarpAligner::writeRealignmentInfo(std::string filename)
{
   if (Dstore.empty())
      return;
   
   std::ofstream os(filename);
   
   os << "Line Duration," << image_params.line_duration << "\n";
   os << "Interline Duration," << image_params.interline_duration << "\n";
   os << "Frame Duration," << image_params.frame_duration << "\n";
   os << "Interframe Duration," << image_params.interframe_duration << "\n";
   os << "Lines," << image_params.n_y << "\n";

   os << "Frame, UnalignedCorrelation, Correlation, Coverage";
   for (int j = 0; j < Dstore[0].size(); j++)
      os << ", p_" << j;
   os << "\n";
   for (int i = 0; i < Dstore.size(); i++)
   {
      os << i << "," << results[i].unaligned_correlation << "," << results[i].correlation << ", " << results[i].coverage;
      for (int j = 0; j < Dstore[i].size(); j++)
         os << ", " << Dstore[i][j].x << std::showpos << Dstore[i][j].y << std::noshowpos << "i";
      os << "\n";
   }
}

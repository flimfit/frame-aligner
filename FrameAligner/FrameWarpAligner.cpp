#include "FrameWarpAligner.h"
#include "Cv3dUtils.h"
#include "Sobel3d.h"
#include "OptimisationModel.h"
#include <functional>
#include <fstream>
#include <algorithm>



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
   
   // Find best starting point
   std::vector<double> starting_point_eval(starting_point.size());
   std::transform(starting_point.begin(), starting_point.end(), starting_point_eval.begin(), [&](auto p)->auto { return model(p); });
   size_t best_start = std::min_element(starting_point_eval.begin(), starting_point_eval.end()) - starting_point_eval.begin();

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
   int i = pi / n_dim;
   int j = pj / n_dim;

   if (j > i) return computeHessianEntry(pj, pi);
   if ((i - j) > 1) return 0;

   auto getV = [this](int i) -> const OffsetVector<float>& {
      int d = i % n_dim;
      int idx = i / n_dim;
      switch (d)
      {
      case 0:
         return VI_dW_dp_x[idx];
      case 1:
         return VI_dW_dp_y[idx];
      case 2:
         return VI_dW_dp_z[idx];
      }
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

   // Diagonal elements
   for (int i = 0; i < nD * n_dim; i++)
      for (int j = 0; j < nD * n_dim; j++)
      {
         double h = computeHessianEntry(i, j);
         H(i, j) = h;
         H(j, i) = h;
      }
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
         jac(i*n_dim) += VI_dW_dp_x[i][p] * err_ptr[p]; // x 
         jac(i*n_dim + 1) += VI_dW_dp_y[i][p] * err_ptr[p]; // y
         if (n_dim == 3)
            jac(i*n_dim + 2) += VI_dW_dp_z[i][p] * err_ptr[p]; // z        
      }
   }
   for (int i = 0; i < (nD - 1); i++)
   {
      int p0 = D_range[i].begin;
      int p1 = D_range[i].end;
      for (int p = p0; p < p1; p++)
      {
         jac(i*n_dim) += VI_dW_dp_x[i][p] * err_ptr[p]; // x
         jac(i*n_dim + 1) += VI_dW_dp_y[i][p] * err_ptr[p]; // y
         if (n_dim == 3)
            jac(i*n_dim + 2) += VI_dW_dp_z[i][p] * err_ptr[p]; // z
         
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

cv::Mat FrameWarpAligner::realignAsFrame(int frame_t, const cv::Mat& frame)
{
   cv::Mat warped;
   warpImage(frame, warped, Dstore[frame_t]);
   return warped;
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

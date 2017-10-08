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
   warper = std::make_shared<CPUFrameWarper>();
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

   warper->setReference(reference, nD, image_params);

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

   auto model = OptimisationModel(warper, frame, raw_frame);

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
      warper->warpImageIntensityPreserving(raw_frame, intensity_preserving, D);
   r.realigned_preserving = reshapeForOutput(intensity_preserving);
   r.done = true;
   
   results[frame_t] = r;

   return r;


}

void FrameWarpAligner::shiftPixel(int frame_t, double& x, double& y, double& z)
{
   if (!realign_params.use_realignment())
      return;

   assert(Dstore.size() > frame_t);

   auto& D = Dstore[frame_t];
   cv::Point3d loc = warper->warpPoint(D, x, y, z);

   x -= loc.x;
   y -= loc.y;
   z -= loc.z;
}



cv::Mat FrameWarpAligner::realignAsFrame(int frame_t, const cv::Mat& frame)
{
   cv::Mat warped;
   warper->warpImage(frame, warped, Dstore[frame_t]);
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

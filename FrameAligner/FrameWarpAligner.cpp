#include "FrameWarpAligner.h"
#include "Cv3dUtils.h"
#include "Sobel3d.h"
#include "OptimisationModel.h"
#include <functional>
#include <fstream>
#include <algorithm>

const bool FrameWarpAligner::supress_hotspots = false;


FrameWarpAligner::FrameWarpAligner(RealignmentParameters params)
{
   realign_params = params;
   if (params.prefer_gpu && GpuFrameWarper::hasSupportedGpu())
      warper = std::make_shared<GpuFrameWarper>();
   else
      warper = std::make_shared<CpuFrameWarper>();
}

cv::Mat FrameWarpAligner::reshapeForOutput(cv::Mat& m, int type)
{
   cv::Mat out;
   if ((dims[Z] == 1) && output2d)
      out = m.reshape(0, 2, &dims[Y]);
   else
      out = m;
   out.convertTo(out, type);
   return out;
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
   if (frame >= results.size()) return false;
   return (results[frame].done);
}

void FrameWarpAligner::setNumberOfFrames(int n_frames_)
{
   n_frames = n_frames_;

   Dstore.clear();
   results.clear();
   
   Dstore.resize(n_frames);
   results.resize(n_frames);
}


void FrameWarpAligner::clearTemp()
{
   warper->clearTemp();
}

void FrameWarpAligner::setReference(int frame_t, const cv::Mat& reference_)
{
   output2d = (reference_.dims == 2);

   dims = {image_params.n_z, image_params.n_y, image_params.n_x};
   n_dim = (image_params.n_z > 1) ? 3 : 2;

   phase_downsampling = realign_params.spatial_binning;

   reference_.copyTo(reference);
   reference.convertTo(reference, CV_32F);
   reference = reshapeForProcessing(reference);
   smoothStack(reference, smoothed_reference);

   cv::Mat reference_downsampled = downsample(reference, phase_downsampling);
   auto downsampled_size = reference_downsampled.size;

   phase_correlator = std::unique_ptr<VolumePhaseCorrelator>(new VolumePhaseCorrelator(downsampled_size[Z], downsampled_size[Y], downsampled_size[X]));


   phase_correlator->setReference((float*) reference_downsampled.data);

   nD = realign_params.n_resampling_points * image_params.n_z + 1;

   warper->setReference(smoothed_reference, nD, image_params);
   
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
   raw_frame_.convertTo(raw_frame, CV_32F);

   raw_frame = reshapeForProcessing(raw_frame);

   if (supress_hotspots)
      for (int z = 0; z < dims[Z]; z++)
      {
         cv::Mat slice = extractSlice(raw_frame, z);
         cv::Mat ref_slice = extractSlice(reference, z);

         double mn, mx;
         cv::minMaxIdx(ref_slice, &mn, &mx);
         for (int i = 0; i < dims[X] * dims[Y]; i++)
            if (slice.at<float>(i) > mx)
               slice.at<float>(i) = 0;
      }


   smoothStack(raw_frame, frame);

   warper->registerFrame(frame);
   auto model = OptimisationModel(warper, frame);

   std::vector<column_vector> starting_point(3, column_vector(nD * n_dim));

   // zero starting point
   std::fill(starting_point[0].begin(), starting_point[0].end(), 0);

   std::vector<cv::Point3d> D(nD, cv::Point3d(0,0,0));
   // last starting point
   //if (frame_t < n_frames)
   //   interpolatePoint3d(Dstore[frame_t], D);
   //D2col(D, starting_point[1], n_dim);

   // rigid starting point

   cv::Mat downsampled = downsample(raw_frame, phase_downsampling);
   cv::Mat zframe = downsampled.clone();
   std::vector<cv::Point3d> D_rigid_z(dims[Z]);
   std::vector<cv::Point3d> D_rigid(nD);

   int idx = 0;
   for (int z = 0; z < dims[Z]; z++)
   {
      zframe.setTo(0);
      for(int zf = std::max(0, z - 2); zf <= std::min(dims[Z], z + 2); zf++)
         extractSlice(downsampled, zf).copyTo(extractSlice(zframe, zf));
      cv::Point3d rigid_shift = phase_correlator->computeShift((float*)zframe.data);
      rigid_shift.x *= phase_downsampling;
      rigid_shift.y *= phase_downsampling;
      D_rigid_z[z] = rigid_shift;

      int n = realign_params.n_resampling_points + (z == 0);
      for (int i = 0; i < n; i++)
         D_rigid[idx++] = rigid_shift;
   }
   D2col(D_rigid, starting_point[1], n_dim);
   std::fill(starting_point[1].begin(), starting_point[1].end(), 0);


   cv::Point3d rigid_shift = phase_correlator->computeShift((float*)downsampled.data);
   std::vector<cv::Point3d> D_rigid_volume(nD, rigid_shift);
   D2col(D_rigid_volume, starting_point[2], n_dim);


   // Find best starting point
   std::vector<double> starting_point_eval(starting_point.size());
   std::transform(starting_point.begin(), starting_point.end(), starting_point_eval.begin(), [&](auto p)->auto { return model(p); });
   size_t best_start = std::min_element(starting_point_eval.begin(), starting_point_eval.end()) - starting_point_eval.begin();

   column_vector x = starting_point[best_start];
   
   
   if (realign_params.type == RealignmentType::Warp)
   {
      try
      {

         find_min_trust_region(dlib::objective_delta_stop_strategy(1e-2),
            model,
            x,
            40 // initial trust region radius
         );
      }
      catch (dlib::error e)
      {
         std::cout << e.info;
      }
   }
   
   
   col2D(x, D, n_dim);
   Dstore[frame_t] = D;
   Dlast = *(D.end() - 2);

   cv::Mat warped_smoothed, warped, mask, intensity_preserving, m;
   warper->warpImage(frame, warped_smoothed, D);
         
   warper->deregisterFrame(frame);
   warper->registerFrame(raw_frame);

   warper->warpImage(raw_frame, warped, D);
   warper->warpImageIntensityPreserving(raw_frame, intensity_preserving, mask, D);
   
   warper->deregisterFrame(raw_frame);
   
   cv::compare(mask, 0, m, cv::CMP_GT);

   
   std::unique_lock<std::mutex> lk(align_mutex);

   RealignmentResult r;
   r.frame = reshapeForOutput(raw_frame, CV_8U);
   r.realigned = reshapeForOutput(warped, CV_8U);
   r.mask = reshapeForOutput(mask, CV_8U);
   r.correlation = correlation(warped_smoothed, smoothed_reference, m);
   r.unaligned_correlation = correlation(frame, smoothed_reference, m);
   r.coverage = ((double)cv::countNonZero(mask)) / (dims[X] * dims[Y] * dims[Z]);
   
   if (r.correlation < realign_params.correlation_threshold || r.coverage < realign_params.coverage_threshold)
      intensity_preserving = 0;

   r.realigned_preserving = reshapeForOutput(intensity_preserving, CV_8U);
   r.done = true;
   
   results[frame_t] = r;
   lk.unlock();

   align_cv.notify_all();

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
   std::unique_lock<std::mutex> lk(align_mutex);
   align_cv.wait(lk, [&] { return results[frame_t].done; });

   cv::Mat warped;
   warper->registerFrame(frame);
   warper->warpImage(frame, warped, Dstore[frame_t]);
   warper->deregisterFrame(frame);
   return warped;
}


void FrameWarpAligner::writeRealignmentInfo(std::string filename)
{
   if (Dstore.empty())
      return;
   
   std::ofstream os(filename);
   
   os << "Realignment Information Version, 2\n";
   os << "Line Duration," << image_params.line_duration << "\n";
   os << "Interline Duration," << image_params.interline_duration << "\n";
   os << "Frame Duration," << image_params.frame_duration << "\n";
   os << "Interframe Duration," << image_params.interframe_duration << "\n";
   os << "Lines," << image_params.n_y << "\n";
   os << "Frames Per Stack," << image_params.n_z << "\n";

   os << "Frame, UnalignedCorrelation, Correlation, Coverage";
   for (int j = 0; j < Dstore[0].size(); j++)
      os << ", dx_" << j << ", dy_" << j << ", dz_" << j;
   os << "\n";
   for (int i = 0; i < Dstore.size(); i++)
   {
      os << i << "," << results[i].unaligned_correlation << "," << results[i].correlation << ", " << results[i].coverage;
      for (int j = 0; j < Dstore[i].size(); j++)
         os << ", " << Dstore[i][j].x << ", " << Dstore[i][j].y << ", " << Dstore[i][j].z;
      os << "\n";
   }
}

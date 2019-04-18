#include "FrameWarpAligner.h"
#include "Cv3dUtils.h"
#include "Sobel3d.h"
#include "OptimisationModel.h"
#include <functional>
#include <fstream>
#include <algorithm>
#include "GpuFrameWarper.h"
#include "Cache_impl.h"

const bool FrameWarpAligner::supress_hotspots = false;


FrameWarpAligner::FrameWarpAligner(RealignmentParameters params)
{
   realign_params = params;
   if (params.prefer_gpu && GpuFrameWarper::getGpuSupportInformation())
   {
#ifdef USE_CUDA_REALIGNMENT
      warper = std::make_shared<GpuFrameWarper>();
#endif
   }
   else
   {
      warper = std::make_shared<CpuFrameWarper>();
   }
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
   smoothed_reference += 1.0f;

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


void FrameWarpAligner::addFrame(int frame_t, CachedMat raw_frame_cache)
{
   if (reference.empty()) throw std::runtime_error("Reference imaging not yet set");

   cv::Mat raw_frame_, raw_frame, frame;
   raw_frame_ = raw_frame_cache->get();
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
   frame += 1.0f;

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

   /*
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
   */
   std::fill(starting_point[1].begin(), starting_point[1].end(), 0);


   cv::Point3d rigid_shift = phase_correlator->computeShift((float*)downsampled.data);
   rigid_shift.x *= phase_downsampling;
   rigid_shift.y *= phase_downsampling;
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

         find_min_trust_region(dlib::objective_delta_stop_strategy(1e-8),
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

   cv::Mat warped_smoothed;
   warper->warpImageInterpolated(frame, warped_smoothed, D);
   warper->deregisterFrame(frame);

   cv::Mat m;
   cv::Mat mask = getMask(raw_frame_cache, D, false);   
   cv::compare(mask, 0, m, cv::CMP_GT);
//   inpaint3d(intensity_preserving, mask, warped);
   
   std::unique_lock<std::mutex> lk(align_mutex);

   auto cache = Cache<cv::Mat>::getInstance();
    

   RealignmentResult r;
   r.frame = raw_frame_cache;
   r.realigned = cache->add(std::bind(&FrameWarpAligner::getInterpolatedWarpedFrame, this, raw_frame_cache, D));
   r.mask = cache->add(std::bind(&FrameWarpAligner::getMask, this, raw_frame_cache, D, true));
   r.realigned_preserving = cache->add(std::bind(&FrameWarpAligner::getWarpedFrame, this, raw_frame_cache, D));

   r.correlation = correlation(warped_smoothed, smoothed_reference, m);
   r.unaligned_correlation = correlation(frame, smoothed_reference, m);
   r.coverage = ((double)cv::countNonZero(mask)) / (dims[X] * dims[Y] * dims[Z]);
   
   r.done = true;
   
   results[frame_t] = r;
   lk.unlock();

   align_cv.notify_all();
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



cv::Mat FrameWarpAligner::realignAsFrame(int frame_t, const cv::Mat& frame, bool interpolate_missing)
{
   std::unique_lock<std::mutex> lk(align_mutex);
   align_cv.wait(lk, [&] { return results[frame_t].done; });

   cv::Mat warped;
   warper->registerFrame(frame);
   if (!results[frame_t].useFrame(realign_params))
      warped = cv::Mat(frame.size(), CV_16U, cv::Scalar(0));
   if (interpolate_missing)
      warper->warpImageInterpolated(frame, warped, Dstore[frame_t]);
   else
      warper->warpImageNormalised(frame, warped, Dstore[frame_t]);
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


cv::Mat FrameWarpAligner::getWarpedFrame(CachedMat frame_, std::vector<cv::Point3d> D)
{
   cv::Mat frame, intensity_preserving, mask;
   frame_->get().convertTo(frame, CV_32F);
   frame = reshapeForProcessing(frame);
   warper->warpImage(frame, intensity_preserving, mask, D);
   intensity_preserving /= mask;
   return reshapeForOutput(intensity_preserving, CV_16U);
}

cv::Mat FrameWarpAligner::getInterpolatedWarpedFrame(CachedMat frame_, std::vector<cv::Point3d> D)
{
   cv::Mat frame, warped;
   frame_->get().convertTo(frame, CV_32F);
   frame = reshapeForProcessing(frame);
   warper->registerFrame(frame);
   warper->warpImageInterpolated(frame, warped, D);
   warper->deregisterFrame(frame);

   return reshapeForOutput(warped, CV_16U);
}

cv::Mat FrameWarpAligner::getMask(CachedMat frame_, std::vector<cv::Point3d> D, bool reshape_for_output)
{
   cv::Mat frame, intensity_preserving, mask;
   frame_->get().convertTo(frame, CV_32F);
   frame = reshapeForProcessing(frame);
   warper->warpImage(frame, intensity_preserving, mask, D);
   return reshape_for_output ? reshapeForOutput(mask, CV_32F) : mask;
}

#include "FrameWarper.h"
#include "Sobel3d.h"

double CpuFrameWarper::computeErrorImage(cv::Mat& wimg, cv::Mat& error_img)
{
   int n_px = dims[X] * dims[Y] * dims[Z];

   error_img = wimg - reference;
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

void CpuFrameWarper::warpImage(const cv::Mat& img, cv::Mat& wimg, const std::vector<cv::Point3d>& D, int invalid_value)
{
   wimg = cv::Mat(dims, CV_32F, cv::Scalar(invalid_value));
   
   for (int z = 0; z < dims[Z]; z++)
      for (int y = 0; y < dims[Y]; y++)
         for (int x = 0; x < dims[X]; x++)
         {

            cv::Point3d loc = warpPoint(D, x, y, z);
            loc += cv::Point3d(x, y, z);

            cv::Point3i loc0(floor(loc.x), floor(loc.y), floor(loc.z));
            cv::Point3d locf(loc.x - loc0.x, loc.y - loc0.y, loc.z - loc0.z);

            // Clamp values slightly outside range
            if ((loc.x < 0) && (loc.x > -1)) loc.x = 0;
            if ((loc.y < 0) && (loc.y > -1)) loc.y = 0;
            if ((loc.z < 0) && (loc.z > -1)) loc.z = 0;

            if ((loc.x > (dims[X] - 1)) && (loc.x < dims[X])) loc.x = dims[X]-1;
            if ((loc.y > (dims[Y] - 1)) && (loc.y < dims[Y])) loc.y = dims[Y]-1;
            if ((loc.z > (dims[Z] - 1)) && (loc.z < dims[Z])) loc.z = dims[Z]-1;

            cv::Point3i loc1 = loc0;
            if (loc1.x < (dims[X]-1)) loc1.x++;
            if (loc1.y < (dims[Y]-1)) loc1.y++;
            if (loc1.z < (dims[Z]-1)) loc1.z++;
            
            if (isValidPoint(loc0, dims))
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

void AbstractFrameWarper::warpImageIntensityPreserving(const cv::Mat& img, cv::Mat& wimg, cv::Mat& coverage, const std::vector<cv::Point3d>& D)

{
   wimg = cv::Mat(dims, CV_8U, cv::Scalar(0));
   coverage = cv::Mat(dims, CV_8U, cv::Scalar(0));
   
   for (int z = 0; z < dims[Z]; z++)
      for (int y = 0; y < dims[Y]; y++)
         for (int x = 0; x < dims[X]; x++)
         {
            cv::Point3d loc = warpPoint(D, x, y, z);

            cv::Vec<int,3> locr = { (int)round(z - loc.z),
                                    (int)round(y - loc.y),
                                    (int)round(x - loc.x) };              
               
            if (isValidPoint(locr, dims))
            {
               wimg.at<uint8_t>(locr) += img.at<float>(z, y, x);
               coverage.at<uint8_t>(locr)++;               
            }
         }
}

void AbstractFrameWarper::computeJacobian(const cv::Mat& error_img, column_vector& jac)
{
   jac.set_size(nD * n_dim);
   std::fill(jac.begin(), jac.end(), 0);

   // we are ignoring stride here, ok as set up
   float* err_ptr = reinterpret_cast<float*>(error_img.data);

   for (int i = 0; i < nD; i++)
   {
      int p0 = VI_dW_dp[i].first();
      int p1 = VI_dW_dp[i].last();
      for (int p = p0; p < p1; p++)
      {
         jac(i*n_dim) += VI_dW_dp[i][p].x * err_ptr[p]; // x 
         jac(i*n_dim + 1) += VI_dW_dp[i][p].y * err_ptr[p]; // y
         if (n_dim == 3)
            jac(i*n_dim + 2) += VI_dW_dp[i][p].z * err_ptr[p]; // z        
      }
   }
}

double CpuFrameWarper::getError(const cv::Mat& frame, const std::vector<cv::Point3d>& D)
{
   cv::Mat warped_image, error_image;
   warpImage(frame, warped_image, D, -1);
   double err = computeErrorImage(warped_image, error_image);
   error_buffer[frame.data] = error_image;
   return err;
}  

void CpuFrameWarper::getJacobian(const cv::Mat& frame, const std::vector<cv::Point3d>& D, column_vector& jac)
{
   cv::Mat error_image = error_buffer[frame.data];
   computeJacobian(error_image, jac);
}



void AbstractFrameWarper::setReference(const cv::Mat& reference_, int nD_, const ImageScanParameters& image_params_)
{
   if (reference_.dims != 3)
      throw std::runtime_error("Reference must be three dimensional");

   reference = reference_;
   nD = nD_;
   image_params = image_params_;
   dims.resize(3);
   for(int i=0; i<reference.dims; i++)
      dims[i] = reference.size[i];

   n_dim = (dims[Z] > 1) ? 3 : 2;

   precomputeInterp(image_params);
   computeSteepestDecentImages(reference);
   computeHessian();

   setupReferenceInformation();
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


void AbstractFrameWarper::precomputeInterp(const ImageScanParameters& image_params)
{
   double pixel_duration = image_params.pixel_duration;
   double frame_duration = image_params.frame_duration;
   double interframe_duration = image_params.interframe_duration;
   double interline_duration = image_params.interline_duration;
   double stack_duration = dims[Z] * interframe_duration;

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
            double t = z * interframe_duration + y * interline_duration + x * pixel_duration;
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

   if (i != (D_range.size() - 2))
      throw std::runtime_error("Something went wrong");

   D_range[i].end = dims[X]*dims[Y]*dims[Z] - 1;

   VI_dW_dp.clear();
   
   for (int i = 0; i < nD; i++)
   {
      int i0 = D_range[std::max(0, i - 1)].begin;
      int i1 = D_range[std::min(i, nD - 2)].end;
      VI_dW_dp.push_back(OffsetVector<cv::Point3f>(i0, i1));
   }
}


void AbstractFrameWarper::computeSteepestDecentImages(const cv::Mat& frame)
{
   // Evaluate gradient of reference

   cv::Mat nabla_Tx(dims, CV_32F), nabla_Ty(dims, CV_32F), nabla_Tz(dims, CV_32F);

   sobel3d(frame, nabla_Tx, nabla_Ty, nabla_Tz);

   float* nabla_Txd = (float*)nabla_Tx.data;
   float* nabla_Tyd = (float*)nabla_Ty.data;
   float* nabla_Tzd = (float*)nabla_Tz.data;
   float* Dfd = (float*)Df.data;

   for (int i = 1; i < nD; i++)
   {
      int p0 = D_range[i - 1].begin;
      int p1 = D_range[i - 1].end;
      for (int p = p0; p < p1; p++)
      {
         cv::Point3f& dp = VI_dW_dp[i][p];

         double jac = Dfd[p];
         dp.x = nabla_Txd[p] * jac;
         dp.y = nabla_Tyd[p] * jac;

         if (n_dim == 3)
            dp.z = nabla_Tzd[p] * jac;
      }
   }
   
   for(int i = 0; i < (nD - 1); i++)
   {
      int p0 = D_range[i].begin;
      int p1 = D_range[i].end;
      for (int p = p0; p < p1; p++)
      {
         cv::Point3f& dp = VI_dW_dp[i][p];
         
         double jac = 1 - Dfd[p];
         dp.x = nabla_Txd[p] * jac;
         dp.y = nabla_Tyd[p] * jac;

         if (n_dim == 3)
            dp.z = nabla_Tzd[p] * jac;
      }
   }
}

double AbstractFrameWarper::computeHessianEntry(int pi, int pj)
{
   int i = pi / n_dim;
   int j = pj / n_dim;

   int di = pi % n_dim;
   int dj = pj % n_dim;


   if (j > i) return computeHessianEntry(pj, pi);
   if ((i - j) > 1) return 0;


   auto get = [](const OffsetVector<cv::Point3f>& v, int p, int d) -> float
   { 
      switch (d)
      {
      case 0: return v.at(p).x;
      case 1: return v.at(p).y;
      case 2: return v.at(p).z;
      }
   };

   auto& vi = VI_dW_dp[i];
   auto& vj = VI_dW_dp[j];
  
   int p0 = std::max(vi.first(), vj.first());
   int p1 = std::min(vi.last(), vj.last());

   double h = 0;
   for (int p = p0; p < p1; p++)
      h += get(vi, p, di) * get(vj, p, dj);
      
   return h;
}

void AbstractFrameWarper::computeHessian()
{   
   H.set_size(n_dim * nD, n_dim * nD);

   for (int i = 0; i < nD * n_dim; i++)
      for (int j = 0; j <= i; j++)
      {
         double h = computeHessianEntry(i, j);
         H(i, j) = h;
         H(j, i) = h;
      }
}

cv::Point3d AbstractFrameWarper::warpPoint(const std::vector<cv::Point3d>& D, int x, int y, int z, int spatial_binning)
{
   double factor = 1.0 / spatial_binning;
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

   cv::Point3d p = f * D[i + 1] + (1 - f) * D[i];
   p *= factor;

   return p;

}
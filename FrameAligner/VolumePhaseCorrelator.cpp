#include "VolumePhaseCorrelator.h"
#include <algorithm>

#define PI 3.14159265358979323846
#define Z 0
#define Y 1
#define X 2


CorrelatorPlan::CorrelatorPlan(const std::vector<int>& dims)
{
   int n_el = dims[Z] * dims[Y] * dims[X];
   int n_complex = dims[Z] * dims[Y] * (dims[X] / 2 + 1);

   in.resize(n_el);
   out.resize(n_complex);

   plan = fftw_plan_dft_r2c(3, dims.data(), in.data(), (fftw_complex*) out.data(), FFTW_MEASURE);
   inv_plan = fftw_plan_dft_c2r(3, dims.data(), (fftw_complex*) out.data(), in.data(), FFTW_MEASURE);
}

CorrelatorPlan::~CorrelatorPlan()
{
   fftw_destroy_plan(plan);
   fftw_destroy_plan(inv_plan);
}


VolumePhaseCorrelator::VolumePhaseCorrelator(int n_z, int n_y, int n_x) : 
   dims({n_z, n_y, n_x}),
   pool(dims)
{
   n_el = dims[X] * dims[Y] * dims[Z];
   n_complex = dims[Z] * dims[Y] * (dims[X] / 2 + 1);

   reference.resize(n_complex);
   window.resize(n_el);

   computeWindow();
}

void VolumePhaseCorrelator::setReference(const float* volume)
{
   auto p = pool.get();

   // Apply window
   for(int i=0; i<n_el; i++)
      p->in[i] = volume[i] * window[i];
   
   fftw_execute(p->plan);

   for(int i=0; i<n_complex; i++)
      reference[i] = conj(p->out[i]);
}


cv::Point3d VolumePhaseCorrelator::computeShift(const float* volume)
{
   auto p = pool.get();
   
   // Apply window
   for(int i=0; i<n_el; i++)
      p->in[i] = volume[i] * window[i];

   // Execute FFT
   fftw_execute(p->plan);

   double eps = std::numeric_limits<double>::min();
   // Compute cross-power spectrum
   for(int i=0; i<n_complex; i++)
   {
      p->out[i] *= reference[i];
      std::complex<double> norm = std::abs(p->out[i])+eps;
      p->out[i] /= (norm);
   }

   // Inverse FFT
   fftw_execute(p->inv_plan);

   // Find peak
   double pk = -std::numeric_limits<double>::max();
   int idx = 0;
   for(int i=0; i<n_el; i++)
   {
      if (p->in[i] > pk)
      {
         pk = p->in[i];
         idx = i;
      } 
   }


   // Compute peak index
   int x = idx % dims[X];
   int y = (idx / dims[X]) % dims[Y];
   int z = idx / (dims[X] * dims[Y]);

   if (x > dims[X]/2) 
      x -= dims[X];
   if (y > dims[Y]/2) 
      y -= dims[Y];
   if (z > dims[Z]/2) 
      z -= dims[Z];


   // Compute subpixel interpolation of peak centre
   double xw = 0, yw = 0, zw = 0, w = 0;
   const int c = 1;
   for (int dz = -c; dz <= c; dz++)
      for (int dy = -c; dy <= c; dy++)
         for (int dx = -c; dx < c; dx++)
         {
            int x1 = x + dx;
            int y1 = y + dy;
            int z1 = z + dz;

            if (x1 < 0) x1 += dims[X];
            if (y1 < 0) y1 += dims[Y];
            if (z1 < 0) z1 += dims[Z];

            if ((x1 >= 0) && (x1 < dims[X]) && 
                (y1 >= 0) && (y1 < dims[Y]) &&
                (z1 >= 0) && (z1 < dims[Z]))
            {
               int idx = x1 + y1 * dims[X] + z1 * dims[X] * dims[Y];
               double v = p->in[idx];
               xw += dx * v;
               yw += dy * v;
               zw += dz * v;
               w += v;
            }
         }
   
   double xf = x;// +xw / w;
   double yf = y;// +yw / w;
   double zf = z;// +zw / w;

   return cv::Point3d(xf, yf, zf);
}

void VolumePhaseCorrelator::computeWindow()
{
   auto windowZ = std::vector<double>(dims[Z], 1); //hann(dims[Z]);
   auto windowY = hann(dims[Y]);
   auto windowX = hann(dims[X]);
   
   int idx = 0;
   for(int z=0; z<dims[Z]; z++)
      for(int y=0; y<dims[Y]; y++)
         for(int x=0; x<dims[X]; x++)
            window[idx++] = windowZ[z]*windowY[y]*windowX[x];
   
}

std::vector<double> VolumePhaseCorrelator::hann(int n)
{
   std::vector<double> window(n, 1);

   if (n> 1)
   {
      double f = 2*PI / (n-1);
      for(int i=0; i<n; i++)
         window[i] = 0.5 * (1 - cos(f * i));
   }

   return window;   
}

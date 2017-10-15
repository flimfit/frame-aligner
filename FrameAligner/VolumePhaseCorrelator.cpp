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
   double pk = 0;
   int idx = 0;
   for(int i=0; i<n_el; i++)
   {
      if (p->in[i] > pk)
      {
         pk = p->in[i];
         idx = i;
      } 
   }

   // Compute peak index - could interpolate here
   int x = idx % dims[X];
   int y = (idx / dims[X]) % dims[Y];
   int z = (idx / (dims[X] * dims[Y])) % dims[Z];

   if (x > dims[X]/2) x -= dims[X];
   if (y > dims[Y]/2) y -= dims[Y];
   if (z > dims[Z]/2) z -= dims[Z];
   
   return cv::Point3d(x, y, z);
}

void VolumePhaseCorrelator::computeWindow()
{
   auto windowZ = hann(dims[Z]);
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
   std::vector<double> window(n);

   if (n == 1)
   {
      window[0] = 1;      
   }
   else
   {
      double f = 2*PI / (n-1);
      for(int i=0; i<n; i++)
         window[i] = 0.5 * (1 - cos(f * i));
   }

   return window;   
}

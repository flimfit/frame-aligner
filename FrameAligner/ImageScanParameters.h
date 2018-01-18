#pragma once
#include <algorithm>

class ImageScanParameters
{
public:
   ImageScanParameters(double line_duration_ = 100, double interline_duration_ = 101, double interframe_duration_ = 102, int n_x_ = 1, int n_y_ = 1, int n_z_ = 1, bool bidirectional = false) :
      line_duration(line_duration_), interline_duration(interline_duration_), interframe_duration(interframe_duration_), n_x(n_x_), n_y(n_y_), n_z(n_z_)
   {
      n_x = std::max(n_x, 1);
      n_y = std::max(n_y, 1);

      if (interline_duration < line_duration)
         interline_duration = 1.01 * line_duration;

      pixel_duration = line_duration / n_x;
      frame_duration = n_y * interline_duration;

      stack_duration = (n_z - 1) * interframe_duration + frame_duration;
   }

   double line_duration;
   double interline_duration;
   double pixel_duration;
   double frame_duration;
   double interframe_duration;
   double stack_duration;
   int n_x;
   int n_y;
   int n_z;
   bool bidirectional;
};

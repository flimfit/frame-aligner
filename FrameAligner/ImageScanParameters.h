#pragma once
#include <algorithm>

class ImageScanParameters
{
public:
   ImageScanParameters(double line_duration = 100, double interline_duration = 101, double interframe_duration = 102, int n_x = 1, int n_y = 1, int n_z = 1, bool bidirectional = false) :
      line_duration(line_duration), interline_duration(interline_duration), interframe_duration(interframe_duration), n_x(n_x), n_y(n_y), n_z(n_z)
   {
      n_x = std::max(n_x, 1);
      n_y = std::max(n_y, 1);

      if (interline_duration < line_duration)
         interline_duration = 1.01 * line_duration;

      pixel_duration = line_duration / n_x;
      frame_duration = n_y * interline_duration;
   }

   double line_duration;
   double interline_duration;
   double pixel_duration;
   double frame_duration;
   double interframe_duration;
   int n_x;
   int n_y;
   int n_z;
   bool bidirectional;
};

#pragma once

namespace Wings
{

namespace DefaultValues
{
  // in degrees
  const double small_angle = 3.0 * (3.14159265358979323846/180.);
  const double small_number_geometry = 1e-3;
  // to compare with balhoff's solutions
  const double small_number_balhoff = 3e-3;
  const double small_number = 1e-10;
  const int n_time_step_digits = 3;
  const int n_processor_digits = 3;
}


}  // end wings

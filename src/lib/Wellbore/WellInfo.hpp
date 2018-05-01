#pragma once

#include <vector>
#include <deal.II/base/point.h>

/*
 * This structure stores info that is necessary to initiate wells
 */

namespace Wings
{

namespace Wellbore
{
// using namespace dealii;


template<int dim>
struct WellInfo
{
  WellInfo(const double                            radius,
           const std::vector<dealii::Point<dim>> & locations,
           const std::string                       name = "");

  double                          radius;
  std::vector<dealii::Point<dim>> locations;
  std::string                     name;
};



template <int dim>
WellInfo<dim>::WellInfo(const double                            radius,
                        const std::vector<dealii::Point<dim>> & locations,
                        const std::string                       name)
    :
    radius(radius),
    locations(locations),
    name(name)
{}



// template<int dim>
// void
// WellInfo<dim>::set_info()
// {
//   this->name = name;
//   this->radius = radius;
//   this->locations = locations;
// }  // end set_info


} // end wellbore

} // end wings

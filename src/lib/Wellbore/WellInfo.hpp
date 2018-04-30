#pragma once

#include <deal.II/base/point.h>

namespace Wings
{

namespace Wellbore
{
// using namespace dealii;

template<int dim>
struct WellInfo
{
  WellInfo();
  // WellInfo(const double                           radius,
  //          // const std::vector<dealii:Point<dim>> & locations,
  //          const std::vector<dealii:Point<dim>> & locations,
  //          const std::string                      name = "");

  // void set_info(const double                           radius,
  //               const std::vector<dealii:Point<dim>> & locations,
  //               const std::string                      name = "");

  std::string             name;
  double                  radius;
  std::vector<dealii::Point<dim>> locations;
};



template <int dim>
WellInfo<dim>::WellInfo()
{}



// template <int dim>
// WellInfo<dim>::WellInfo(const double                           radius,
//                         // const std::vector<dealii:Point<dim>> & locations,
//                         const std::vector<dealii:Point<dim>> & locations,
//                         const std::string                      name = "")
//     :
//     radius(radius),
//     locations(locations),
//     name(name)
// {}



// template<int dim>
// void
// WellInfo<dim>::set_info()
// {
//   this->name = name;
//   this->radius = radius;
//   this->locations = locations;
// }  // end set_info


} // end welbore

} // end wings

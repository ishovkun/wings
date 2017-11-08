#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <Schedule.cc>


namespace Wellbore
{
	using namespace dealii;

	template <int dim>
	class Wellbore : public Function<dim>
  {
  public:
    Wellbore(const std::vector< Point<dim> >& locations_,
             const int                        direction_,
             const double                     radius_);
    void set_control(const Schedule::WellControl& control_);
    Schedule::WellControl get_control();

  private:
    std::vector< Point<dim> > locations;
    int                       direction;
    double                    radius;
    Schedule::WellControl     control;
  };  // eom


  template <int dim>
  Wellbore<dim>::Wellbore(const std::vector< Point<dim> >& locations_,
                          const int                        direction_,
                          const double                     radius_)
    :
    locations(locations_),
    direction(direction_),
    radius(radius_)
  {} //  eom


  template <int dim>
  void Wellbore<dim>::set_control(const Schedule::WellControl& control_)
  {
    control = control_;
  }


  template <int dim>
  Schedule::WellControl
  Wellbore<dim>::get_control()
  {
    return control;
  }
}  // end of namespace

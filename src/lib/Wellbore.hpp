#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

enum WellControl {pressure_control, flow_control_total, flow_control_water,
                  flow_control_oil, flow_control_gas};

namespace Wellbore
{
	using namespace dealii;

	template <int dim>
	class Wellbore : public Function<dim>
  {
    Wellbore(const std::vector< Point<dim> >& locations_,
             const int                        direction_);
    void set_schedule(const WellControl control_, const double value);

  private:
    WellControl               control;
    double                    control_value;
    std::vector< Point<dim> > locations;
    int                       direction;
  };  // eom

  template <int dim>
  Wellbore<dim>::Wellbore(const std::vector< Point<dim> >& locations_,
                          const int                        direction_)
    :
    locations(locations_),
    direction(direction_)
  {} //  eom

}  // end of namespace

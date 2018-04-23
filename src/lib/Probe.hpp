#pragma once

#include <Model.hpp>

namespace Wings {

namespace Probe {


static const int dim = 3;

template <int dim>
using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

struct SolutionValues
{
  SolutionValues(const int n_phases = 3) {saturation.reinit(n_phases);}
  Vector<double> saturation;
  // double div_u, div_old_u;
  double pressure;
  Tensor<2, 3> grad_u, grad_old_u;
};


class Probe
{
 public:
  Probe(const Model<dim> & model);
  ~Probe();

  // fetching data
  double get_biot_coefficient() const;
  double gravity() const;
  double get_rock_compressibility(const Point<dim> &p) const;
  void get_pvt(const double   pressure,
               const int      phase,
               PVTValues    & pvt_values) const;

  double get_total_density() const;
  double get_porosity() const;
  double get_absolute_permeability() const;
  // double get_relative_permeability() const;

  virtual void begin_cell() = 0;
  virtual void next_cell() = 0;
  virtual void set_cell(CellIterator & cell) = 0;

 protected:
  virtual void extract_solution_values() = 0;

  const Model<dim> & model;
  CellIterator  fluid_cell, solid_cell;
};



Probe(const Model & model)
    :
    model(model)
{}



inline double
Probe::get_biot_coefficient() const
{
  AssertThrow(solid_model != SolidModelType::Compressibility,
              ExcMessage("Biot coefficient inapplicable for compressibility model"));

  return model.biot_coefficient;
} //eom



inline double
Probe::get_porosity() const
{
  return model.get_porosity->value(fluid_cell->center());
} // eom



inline TensorFunction<2,dim,double>
Probe::get_permeability() const
{
  return model.get_permeability->value(fluid_cell->center());
} // eom



template<int dim>
double
Probe::get_rock_compressibility() const
{
  if (solid_model == SolidModelType::Compressibility)
  {
    return rock_compressibility_constant;
  }
  else if (solid_model == SolidModelType::Elasticity)
  {
    const double E = model.get_young_modulus->value(p);
    const double nu = model.get_poisson_ratio->value(p);
    const double bulk_modulus = E/3.0/(1.0-2.0*nu);
    const double alpha = get_biot_coefficient();
    const double phi = model.get_porosity->value(p);

    AssertThrow(alpha > phi /* || alpha == 0.0 */,
                ExcMessage("Biot coef should be > porosity"));
    const double rec_N = (alpha - phi) * (1.0 - alpha) / bulk_modulus;
    return rec_N;
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }
}  // end get_rock_compressibility



double
Probe<dim>::get_total_density()
{

}  // end get_total_density



} // end probe namespace

} // end wings

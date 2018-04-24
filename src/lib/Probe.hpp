#pragma once

#include <Model.hpp>

namespace Wings {

namespace Probe {
using namespace dealii;


static const int dim = 3;

template <int dim>
using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;


template<int n_phases>
struct SolutionValues
{
  Tensor<1,n_phases> saturation;
  double pressure;
  Tensor<2, dim> grad_u, grad_old_u;
};


template <int n_phases>
class Probe
{
 public:
  Probe(const Model::Model<dim> & model);
  // ~Probe();

  // fetching data
  double get_biot_coefficient() const;
  double gravity() const;
  double get_rock_compressibility() const;
  // void get_pvt(const double   pressure,
  //              const int      phase,
  //              PVTValues    & pvt_values) const;

  double get_total_density() const;
  double get_porosity() const;
  Tensor<2, dim> get_absolute_permeability() const;
  // double get_relative_permeability() const;

  virtual void begin_cell();
  virtual void next_cell();
  virtual void set_cell(CellIterator<dim> & cell);

 protected:
  // virtual void extract_solution_values() = 0;

  const Model::Model<dim> & model;
  DoFHandler<dim> * fluid_dof_handler;
  DoFHandler<dim> * solid_dof_handler;
  CellIterator<dim>  fluid_cell, solid_cell;
};



template <int n_phases>
Probe<n_phases>::Probe(const Model::Model<dim> & model)
    :
    model(model)
{}



template <int n_phases>
void Probe<n_phases>::begin_cell()
{
  fluid_cell = fluid_dof_handler->begin_active();
  solid_cell = solid_dof_handler->begin_active();
}  // eom



template <int n_phases>
void Probe<n_phases>::next_cell()
{
  fluid_cell++;
  solid_cell++;
}  // eom


template <int n_phases>
void Probe<n_phases>::set_cell(CellIterator<dim> & cell)
{
  fluid_cell =
      DoFHandler<dim>::
      active_cell_iterator(&(fluid_dof_handler->get_triangulation()),
                           cell->level(),
                           cell->index(),
                           fluid_dof_handler);
  solid_cell =
      DoFHandler<dim>::
      active_cell_iterator(&(solid_dof_handler->get_triangulation()),
                           cell->level(),
                           cell->index(),
                           solid_dof_handler);
}  // eom


template <int n_phases>
inline double
Probe<n_phases>::get_biot_coefficient() const
{
  AssertThrow(model.solid_model != Model::SolidModelType::Compressibility,
              ExcMessage("Biot coefficient inapplicable for compressibility model"));

  return model.get_biot_coefficient();
} //eom



template <int n_phases>
inline double
Probe<n_phases>::get_porosity() const
{
  return model.get_porosity->value(fluid_cell->center());
} // eom



template <int n_phases>
inline Tensor<2, dim>
Probe<n_phases>::get_absolute_permeability() const
{
  return model.get_permeability->value(fluid_cell->center());
} // eom



template <int n_phases>
double
Probe<n_phases>::get_rock_compressibility() const
{
  Point<dim> p = fluid_cell->center();

  if (model.solid_model == Model::SolidModelType::Compressibility)
  {
    return model.rock_compressibility_constant;
  }
  else if (model.solid_model == Model::SolidModelType::Elasticity)
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



template <int n_phases>
double
Probe<n_phases>::get_total_density() const
{
  // const double phi_0 = model.get_porosity->value(fluid_cell);
}  // end get_total_density



} // end probe namespace

} // end wings

#pragma once

#include <Model.hpp>
#include <ElasticSolver.hpp>
#include <SolverIMPES.hpp>

namespace Wings {

namespace Probe {
using namespace dealii;


template <int dim>
using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;



template<int dim, int n_phases>
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

  virtual std::vector<CellIterator<dim>> begin_cells();
  virtual CellIterator<dim> endc();
  virtual void set_cell(CellIterator<dim> & cell);

 protected:
  // virtual void extract_solution_values() = 0;

  const Model::Model<dim> & model;
  FluidSolvers::FluidSolverBase<dim,n_phases> * fluid_solver;
  SolidSolvers::SolidSolverBase<dim,n_phases> * solid_solver;
  // DoFHandler<dim> * fluid_dof_handler;
  // DoFHandler<dim> * solid_dof_handler;
  // CellIterator<dim>  fluid_cell, solid_cell;
};



template<int dim, int n_phases>
Probe<dim,n_phases>::Probe(const Model::Model<dim> & model)
    :
    model(model)
{}



template<int dim, int n_phases>
std::vector<CellIterator<dim>>
Probe<dim,n_phases>::begin_cells()
{
  std::vector<CellIterator<dim>> cells(1);

  if (model.solid_model != Model::SolidModelType::Compressibility)
    cells.resize(2);

  cells[0] = fluid_solver->get_dof_handler().begin_active();

  if (model.solid_model != Model::SolidModelType::Compressibility)
    cells[1] = solid_solver->get_dof_handler().begin_active();

  return cells;
}  // eom



template<int dim, int n_phases>
CellIterator<dim> Probe<dim,n_phases>::endc()
{
  return fluid_solver->get_dof_handler().end();
}  // eom

// template<int dim, int n_phases>
// void Probe<dim,n_phases>::next_cell()
// {
//   fluid_cell++;
//   solid_cell++;
// }  // eom


template<int dim, int n_phases>
void Probe<dim,n_phases>::set_cell(CellIterator<dim> & cell)
{
  // fluid_cell =
  //     CellIterator<dim>(&(fluid_dof_handler->get_triangulation()),
  //                       cell->level(),
  //                       cell->index(),
  //                       fluid_dof_handler);
  // solid_cell =
  //     CellIterator<dim>(&(solid_dof_handler->get_triangulation()),
  //                          cell->level(),
  //                          cell->index(),
  //                          solid_dof_handler);
}  // eom


template<int dim, int n_phases>
inline double
Probe<dim,n_phases>::get_biot_coefficient() const
{
  AssertThrow(model.solid_model != Model::SolidModelType::Compressibility,
              ExcMessage("Biot coefficient inapplicable for compressibility model"));

  return model.get_biot_coefficient();
} //eom



template<int dim, int n_phases>
inline double
Probe<dim,n_phases>::get_porosity() const
{
  throw(ExcNotImplemented());
  // return model.get_porosity->value(fluid_cell->center());
} // eom



template<int dim, int n_phases>
inline Tensor<2, dim>
Probe<dim,n_phases>::get_absolute_permeability() const
{

  throw(ExcNotImplemented());
  // return model.get_permeability->value(fluid_cell->center());
} // eom



template<int dim, int n_phases>
double
Probe<dim,n_phases>::get_rock_compressibility() const
{

  throw(ExcNotImplemented());
  // Point<dim> p = fluid_cell->center();

  // if (model.solid_model == Model::SolidModelType::Compressibility)
  // {
  //   return model.rock_compressibility_constant;
  // }
  // else if (model.solid_model == Model::SolidModelType::Elasticity)
  // {
  //   const double E = model.get_young_modulus->value(p);
  //   const double nu = model.get_poisson_ratio->value(p);
  //   const double bulk_modulus = E/3.0/(1.0-2.0*nu);
  //   const double alpha = get_biot_coefficient();
  //   const double phi = model.get_porosity->value(p);

  //   AssertThrow(alpha > phi /* || alpha == 0.0 */,
  //               ExcMessage("Biot coef should be > porosity"));
  //   const double rec_N = (alpha - phi) * (1.0 - alpha) / bulk_modulus;
  //   return rec_N;
  // }
  // else
  // {
  //   AssertThrow(false, ExcNotImplemented());
  // }
}  // end get_rock_compressibility



template<int dim, int n_phases>
double
Probe<dim,n_phases>::get_total_density() const
{
  throw(ExcNotImplemented());
  // const double phi_0 = model.get_porosity->value(fluid_cell);
}  // end get_total_density



} // end probe namespace

} // end wings

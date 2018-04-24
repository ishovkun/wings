#pragma once
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>

// #include <Probe.hpp>
#include <SolutionValues.h>

namespace Wings {

namespace FluidSolvers
{
using namespace dealii;


template <int dim, int n_phases>
class FluidSolverBase
{
 public:
  virtual void setup_dofs() = 0;
  virtual unsigned int solve_time_step(const double time_step) = 0;
  // virtual void revert_to_old_time_step() = 0;
  // save old iter solution for comparison
  // virtual void save_solution() = 0;
  // coupling with solid solver
  virtual void set_coupling(const DoFHandler<dim>               & solid_dof_handler,
                            const TrilinosWrappers::MPI::Vector & displacement,
                            const TrilinosWrappers::MPI::Vector & old_displacement,
                            const FEValuesExtractors::Vector    & extractor) = 0;
  // for output
  virtual void attach_data(DataOut<dim> & data_out) const = 0;

  // accessing private members
  virtual const DoFHandler<dim>    & get_dof_handler() = 0;
  virtual const FiniteElement<dim> & get_fe() = 0;
  // needed for probe class
  virtual void extract_solution_data
  (const typename DoFHandler<dim>::active_cell_iterator & cell,
   SolutionValues<dim,n_phases>                         & solution_values);
};

} // end of namespace

} // end wings

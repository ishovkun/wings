#pragma once
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>

// custom modules
#include <SolutionValues.h>

namespace Wings {

namespace SolidSolvers
{

using namespace dealii;

template <int dim, int n_phases>
class SolidSolverBase
{
 public:
  virtual void setup_dofs() = 0;
  virtual unsigned int solve_time_step(const double time_step) = 0;
  // virtual void revert_to_old_time_step() = 0;
  // save old iter solution for comparison
  // virtual void save_solution() = 0;
  // coupling with solid solver
  virtual void set_coupling(const DoFHandler<dim> & fluid_dof_handler) = 0;
  // for output
  virtual void attach_data(DataOut<dim> & data_out) const = 0;

  // accessing private members
  // const MatrixType         & get_system_matrix() = 0;
  // const VectorType1        & get_rhs_vector() = 0;
  virtual const DoFHandler<dim>    & get_dof_handler() = 0;

  virtual void extract_solution_data
  (const typename DoFHandler<dim>::active_cell_iterator & cell,
   SolutionValues<dim,n_phases>                         & solution_values) = 0;
};

} // end of namespace

} // end wings

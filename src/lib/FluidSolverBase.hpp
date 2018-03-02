#pragma once
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>


namespace FluidSolvers
{
using namespace dealii;


// template <int dim, typename MatrixType, typename VectorType1, typename VectorType2>
template <int dim>
class FluidSolverBase
{
  virtual void setup_dofs() = 0;
  virtual void assemble_system(const double time_step) = 0;
  virtual void solve_time_step() = 0;
  virtual void revert_to_old_time_step() = 0;
  // coupling with solid solver
  virtual void set_coupling(const DoFHandler<dim>               & solid_dof_handler,
                            const TrilinosWrappers::MPI::Vector & displacement,
                            const TrilinosWrappers::MPI::Vector & old_displacement,
                            const FEValuesExtractors::Vector    & extractor) = 0;
  // for output
  virtual void attach_data(DataOut<dim> & data_out) const = 0;

  // accessing private members
  // const MatrixType         & get_system_matrix() = 0;
  // const VectorType1        & get_rhs_vector() = 0;
  const DoFHandler<dim>    & get_dof_handler() = 0;
  const FiniteElement<dim> & get_fe() =0;
};

} // end of namespace

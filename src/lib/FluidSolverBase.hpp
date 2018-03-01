
namespace FluidSolvers
{


template <int dim, typename VectorType, typename MatrixType>
class FluidSolverBase
{
  virtual void setup_dofs() = 0;
  void assemble_system(const double time_step);
  void solve();
  // coupling with solid solver
  void set_coupling(const DoFHandler<dim>               & solid_dof_handler,
                    const VectorType & displacement_vector,
                    const VectorType & old_displacement_vector,
                    const FEValuesExtractors::Vector    & extractor) = 0;
  // for output
  void attach_data(DataOut<dim> & data_out) const = 0;

  // accessing private members
  const MatrixType      & get_system_matrix();
  const VectorType      & get_rhs_vector();
  const DoFHandler<dim> & get_dof_handler();
  const FE_DGQ<dim>     & get_fe();
};

} // end of namespace

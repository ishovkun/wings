#pragma once

#include <deal.II/fe/fe_system.h>

namespace SolidSolvers
{
using namespace dealii;


template <int dim>
class ElasticSolver
{
 public:
  ElasticSolver(MPI_Comm                                  &mpi_communicator,
                    parallel::distributed::Triangulation<dim> &triangulation,
                    const Model::Model<dim>                   &model,
                    ConditionalOStream                        &pcout);
  ~ElasticSolver();
  void set_coupling();
  /* setup degrees of freedom for the current triangulation
   * and allocate memory for solution vectors */
  void setup_dofs();
  // Fill system matrix and rhs vector
  void assemble_system();
  // solve linear system syste_matrix*solution= rhs_vector
  unsigned int solve();
  // accessing private members
  const TrilinosWrappers::SparseMatrix & get_system_matrix();
  const TrilinosWrappers::MPI::Vector  & get_rhs_vector();
  const DoFHandler<dim>                & get_dof_handler();
  const FESystem<dim>                  & get_fe();

 private:
  MPI_Comm                                  & mpi_communicator;
  parallel::distributed::Triangulation<dim> & triangulation;
  DoFHandler<dim>                             dof_handler;
  FESystem<dim>                               fe;
  const Model::Model<dim>                   & model;
  ConditionalOStream                        & pcout;
  // Matrices and vectors
  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector  rhs_vector;

 public:
  // solution vectors
  TrilinosWrappers::MPI::Vector solution, old_solution;
  TrilinosWrappers::MPI::Vector relevant_solution;
  // partitioning
  IndexSet                      locally_owned_dofs, locally_relevant_dofs;
};



template <int dim>
ElasticSolver<dim>::
ElasticSolver(MPI_Comm                                  &mpi_communicator,
                  parallel::distributed::Triangulation<dim> &triangulation,
                  const Model::Model<dim>                   &model,
                  ConditionalOStream                        &pcout)
    :
    mpi_communicator(mpi_communicator),
    triangulation(triangulation),
    dof_handler(triangulation),
    fe(FE_Q<dim>(1), dim), // dim linear shape functions
    model(model),
    pcout(pcout)
{}



template <int dim>
ElasticSolver<dim>::~ElasticSolver()
{
  dof_handler.clear();
} // eom




template <int dim>
void
ElasticSolver<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);

  { // partitioning
    locally_owned_dofs.clear();
    locally_relevant_dofs.clear();
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);
  }
  { // system matrix
    system_matrix.clear();
    TrilinosWrappers::SparsityPattern
        sparsity_pattern(locally_owned_dofs, mpi_communicator);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);
    sparsity_pattern.compress();
    system_matrix.reinit(sparsity_pattern);
  }
  { // vectors
    solution.reinit(locally_owned_dofs, mpi_communicator);
    relevant_solution.reinit(locally_relevant_dofs, mpi_communicator);
    old_solution.reinit(locally_relevant_dofs, mpi_communicator);
    rhs_vector.reinit(locally_owned_dofs, locally_relevant_dofs,
                      mpi_communicator, /* omit-zeros=*/ true);
  }

}  // eom



// template <int dim>
// void
// ElasticSolver<dim>::assmeble_system()
// {
//   QGauss<dim>   pressure_quadrature_formula(1);
//   QGauss<dim>   quadrature_formula(fe.degree() + 1);

//   FEValues<dim> fe_values(fe, quadrature_formula,
//                           update_values | update_gradients |
//                           update_JxW_values);
//   FEValues<dim> fe_values_pressure(pressure_fe, pressure_quadrature_formula,
//                                    update_values);

//   const unsigned int dofs_per_cell = fe.dofs_per_cell;
//   const unsigned int n_q_points = quadrature_formula.size();

//   std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

//   typename DoFHandler<dim>::active_cell_iterator
//       cell = dof_handler.begin_active(),
//       endc = dof_handler.end(),
//       pressure_cell = pressure_dof_handler.begin_active();

//   system_matrix = 0;
//   rhs_vector = 0;

//   for (; cell!=endc; ++cell, ++pressure_cell)
//     if (cell->is_locally_owned())
//     {
//       fe_values.reinit(cell);
//     } // end cell loop
// }  // eom


} // end of namespace

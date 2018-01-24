#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/config.h>  // for numbers::is_nan
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
// Trilinos stuff
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

// Custom modules
#include <Model.hpp>
#include <Wellbore.hpp>
#include <CellValues.hpp>
#include <ExtraFEData.hpp>

namespace FluidSolvers
{
  using namespace dealii;


template <int dim>
class PressureSolver
{
 public:
  // PressureSolver(const Triangulation<dim>  &triangulation,
  //                const Data::DataBase<dim> &data_);
  PressureSolver(MPI_Comm                                  &mpi_communicator_,
                 parallel::distributed::Triangulation<dim> &triangulation_,
                 const Model::Model<dim>                   &model_,
                 ConditionalOStream                        &pcout_);
  ~PressureSolver();

  void setup_dofs();
  void assemble_system(CellValues::CellValuesBase<dim>                  &cell_values,
                       CellValues::CellValuesBase<dim>                  &neighbor_values,
                       const double                                      time_step,
                       const std::vector<TrilinosWrappers::MPI::Vector> &saturation);
  unsigned int solve();
  // accessing private members
  const TrilinosWrappers::SparseMatrix& get_system_matrix();
  const TrilinosWrappers::MPI::Vector&  get_rhs_vector();
  const DoFHandler<dim> &               get_dof_handler();
  const FE_DGQ<dim> &                   get_fe();

 private:
  MPI_Comm                                  &mpi_communicator;
  parallel::distributed::Triangulation<dim> &triangulation;
  DoFHandler<dim>                           dof_handler;
  FE_DGQ<dim>                               fe;
  const Model::Model<dim>                   &model;
  ConditionalOStream                        &pcout;

  // Matrices and vectors
  // TrilinosWrappers::SparsityPattern         sparsity_pattern;
  TrilinosWrappers::SparseMatrix            system_matrix;
  std::vector<IndexSet>                     owned_partitioning;


  // SparsityPattern
  // typedef MeshWorker::DoFInfo<dim> DoFInfo;
  // typedef MeshWorker::IntegrationInfo<dim> CellInfo;

 public:
  TrilinosWrappers::MPI::Vector solution, old_solution, rhs_vector;
  TrilinosWrappers::MPI::Vector relevant_solution;
  IndexSet                      locally_owned_dofs, locally_relevant_dofs;
};


template <int dim>
PressureSolver<dim>::
PressureSolver(MPI_Comm                                  &mpi_communicator_,
               parallel::distributed::Triangulation<dim> &triangulation_,
               const Model::Model<dim>                 &model_,
               ConditionalOStream                        &pcout_)
    :
    mpi_communicator(mpi_communicator_),
    triangulation(triangulation_),
    dof_handler(triangulation_),
    fe(0), // since we want finite volumes
    model(model_),
    pcout(pcout_)
{}  // eom


template <int dim>
PressureSolver<dim>::~PressureSolver()
{
  dof_handler.clear();
}  // eom


template <int dim>
void PressureSolver<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);
  locally_owned_dofs.clear();
  locally_relevant_dofs.clear();
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler,
                                          locally_relevant_dofs);

  { // system matrix
    system_matrix.clear();
    TrilinosWrappers::SparsityPattern
        sparsity_pattern(locally_owned_dofs, mpi_communicator);
    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         sparsity_pattern);
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
} // eom



template <int dim>
void
PressureSolver<dim>::
assemble_system(CellValues::CellValuesBase<dim>                  &cell_values,
                CellValues::CellValuesBase<dim>                  &neighbor_values,
                const double                                      time_step,
                const std::vector<TrilinosWrappers::MPI::Vector> &saturation)
{
  // Only one integration point in FVM
  QGauss<dim>       quadrature_formula(1);
  QGauss<dim-1>     face_quadrature_formula(1);

  FEValues<dim> fe_values(fe, quadrature_formula, update_values);
  FEValues<dim> fe_values_neighbor(fe, quadrature_formula, update_values);

  // the following two objects only get geometry data
  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_normal_vectors);
  // We need JxW flag for subfaces since there is no
  // method to determine sub face area in triangulation class
  FESubfaceValues<dim> fe_subface_values(fe, face_quadrature_formula,
                                         update_normal_vectors |
                                         update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<types::global_dof_index>
      dof_indices(dofs_per_cell),
      dof_indices_neighbor(dofs_per_cell);

  // objects to store local data
  Tensor<1, dim>       dx_ij, normal;
  std::vector<double>  p_values(quadrature_formula.size()),
                       p_old_values(quadrature_formula.size());
  std::vector< std::vector<double> >  s_values(model.n_phases()-1);
  for (auto & c: s_values)
    c.resize(face_quadrature_formula.size());
  // this one stores both saturation values and geomechanics
  std::vector<double>  extra_values(model.n_phases() - 1);

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

  system_matrix = 0;
  rhs_vector = 0;

  for (; cell!=endc; ++cell)
  {
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(old_solution, p_old_values);
      fe_values.get_function_values(relevant_solution, p_values);

      // std::cout << "cell: " << i << std::endl;
      // double p_i = solution[i];
      const double p_old = p_old_values[0];
      const double p = p_values[0];

      for (unsigned int c=0; c<model.n_phases() - 1; ++c)
      {
        fe_values.get_function_values(saturation[c], s_values[c]);
        extra_values[c] = s_values[c][0];
      }

      cell_values.update(cell, p, extra_values,
                         /* update_wells = */ true);

      const double B_ii = cell_values.get_mass_matrix_entry();
      const double J_i = cell_values.J;
      const double Q_i = cell_values.Q;

      // double matrix_ii = B_ii/time_step + J_i;
      double matrix_ii = 0;
      double rhs_i = B_ii/time_step*p_old + Q_i;

      cell->get_dof_indices(dof_indices);
      const unsigned int i = dof_indices[0];
      unsigned int j = 0;
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (cell->at_boundary(f) == false)
        {
          if((cell->neighbor(f)->level() == cell->level() &&
              cell->neighbor(f)->has_children() == false) ||
             cell->neighbor_is_coarser(f))
          {
            const auto & neighbor = cell->neighbor(f);
            fe_values.reinit(neighbor);
            fe_face_values.reinit(cell, f);

            fe_values.get_function_values(relevant_solution, p_values);
            const double p_neighbor = p_values[0];
            for (unsigned int c=0; c<model.n_phases() - 1; ++c)
            {
              fe_values.get_function_values(saturation[c], s_values[c]);
              extra_values[c] = s_values[c][0];
            }

            normal = fe_face_values.normal_vector(0); // 0 is gauss point
            const double dS = cell->face(f)->measure();  // face area
            dx_ij = cell->neighbor(f)->center() - cell->center();

            // assemble local matrix and distribute
            neighbor_values.update(neighbor, p_neighbor, extra_values,
                                   /* update_well = */ false);
            std::cout << "fuck "  << std::endl;
            cell_values.update_face_values(neighbor_values, dx_ij, normal, dS);
            // distribute
            matrix_ii += cell_values.T_face;
            rhs_i += cell_values.G_face;

            neighbor->get_dof_indices(dof_indices_neighbor);
            j = dof_indices_neighbor[0];
            system_matrix.add(i, j, -cell_values.T_face);
          }
          else if ((cell->neighbor(f)->level() == cell->level()) &&
                   (cell->neighbor(f)->has_children() == true))
          {
            for (unsigned int subface=0;
                 subface<cell->face(f)->n_children(); ++subface)
            {
              // compute parameters
              const auto & neighbor
                  = cell->neighbor_child_on_subface(f, subface);

              fe_values.reinit(neighbor);
              fe_face_values.reinit(cell, f);

              fe_values.get_function_values(relevant_solution, p_values);
              const double p_neighbor = p_values[0];
              for (unsigned int c=0; c<model.n_phases() - 1; ++c)
                fe_values.get_function_values(saturation[c], s_values[c]);

              neighbor->get_dof_indices(dof_indices_neighbor);
              j = dof_indices_neighbor[0];
              fe_subface_values.reinit(cell, f, subface);
              normal = fe_subface_values.normal_vector(0); // 0 is gauss point
              const double dS = fe_subface_values.JxW(0);
              dx_ij = neighbor->center() - cell->center();

              // assemble local matrix
              neighbor_values.update(neighbor, p_neighbor, extra_values);
              cell_values.update_face_values(neighbor_values, dx_ij, normal, dS);
              // distribute
              matrix_ii += cell_values.T_face;
              rhs_i += cell_values.G_face;
              system_matrix.add(i, j, -cell_values.T_face);
            }
          } // end case neighbor is finer

        } // end if face not at boundary
      }  // end face loop

      system_matrix.add(i, i, matrix_ii);
      rhs_vector[i] += rhs_i;

      // std::cout << "------------------------------\n";
    } // end local cells
  } // end cells loop
  system_matrix.compress(VectorOperation::add);
  rhs_vector.compress(VectorOperation::add);
} // eom


template <int dim>
unsigned int
PressureSolver<dim>::solve()
{
  double tol = 1e-10*rhs_vector.l2_norm();
  if (tol == 0.0)
    tol = 1e-10;
  SolverControl solver_control(1000, tol);
  TrilinosWrappers::SolverCG::AdditionalData additional_data_cg;
  TrilinosWrappers::SolverCG
      solver(solver_control, additional_data_cg);

  // LA::MPI::PreconditionAMG preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData additional_data_amg;
  TrilinosWrappers::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix, additional_data_amg);
  solver.solve(system_matrix, solution, rhs_vector, preconditioner);
  return solver_control.last_step();
}

// template <int dim>
// void PressureSolver<dim>::print_system_matrix(const double denominator) const
// {
//   // out, precision, scientific
//   system_matrix.print_formatted(std::cout, 1, false, 0, " ", denominator);
// }  // eom


template <int dim>
const TrilinosWrappers::SparseMatrix&
PressureSolver<dim>::get_system_matrix()
{
  return system_matrix;
}  // eom


template <int dim>
const TrilinosWrappers::MPI::Vector&
PressureSolver<dim>::get_rhs_vector()
{
  return rhs_vector;
}  // eom



template <int dim>
const DoFHandler<dim> &
PressureSolver<dim>::get_dof_handler()
{
  return dof_handler;
}  // eom


template <int dim>
const FE_DGQ<dim> &
PressureSolver<dim>::get_fe()
{
  return fe;
}  // eom
}  // end of namespace

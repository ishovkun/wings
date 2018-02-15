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
#include <CellValues/CellValuesBase.hpp>
#include <ExtraFEData.hpp>

namespace FluidSolvers
{
using namespace dealii;


template <int dim>
class PressureSolver
{
 public:
  PressureSolver(MPI_Comm                                  &mpi_communicator_,
                 parallel::distributed::Triangulation<dim> &triangulation_,
                 const Model::Model<dim>                   &model_,
                 ConditionalOStream                        &pcout_);
  ~PressureSolver();
  /* setup degrees of freedom for the current triangulation
   * and allocate memory for solution vectors */
  void setup_dofs();
  // Fill system matrix and rhs vector
  void assemble_system(CellValues::CellValuesBase<dim>                  &cell_values,
                       CellValues::CellValuesBase<dim>                  &neighbor_values,
                       const double                                      time_step,
                       const std::vector<TrilinosWrappers::MPI::Vector> &saturation);
  // solve linear system syste_matrix*solution= rhs_vector
  unsigned int solve();
  void set_coupling(const DoFHandler<dim>               & solid_dof_handler,
                    const TrilinosWrappers::MPI::Vector & displacement_vector);
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
  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector  rhs_vector;

  // Pointers to solid solver objects
  const DoFHandler<dim>               * p_solid_dof_handler;
  const TrilinosWrappers::MPI::Vector * p_displacement_vector;
  bool coupled_with_solid;

 public:
  TrilinosWrappers::MPI::Vector solution, old_solution;
  TrilinosWrappers::MPI::Vector relevant_solution;
  // partitioning
  IndexSet                      locally_owned_dofs, locally_relevant_dofs;
};


template <int dim>
PressureSolver<dim>::
PressureSolver(MPI_Comm                                  &mpi_communicator_,
               parallel::distributed::Triangulation<dim> &triangulation_,
               const Model::Model<dim>                   &model_,
               ConditionalOStream                        &pcout_)
    :
    mpi_communicator(mpi_communicator_),
    triangulation(triangulation_),
    dof_handler(triangulation_),
    fe(0), // since we want finite volumes
    model(model_),
    pcout(pcout_),
    coupled_with_solid(false)
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
    DoFTools::make_flux_sparsity_pattern(dof_handler, sparsity_pattern);
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
  QGauss<dim>   quadrature_formula(1);
  QGauss<dim-1> face_quadrature_formula(1);

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
  Tensor<1, dim>       normal;
  std::vector<double>  p_values(quadrature_formula.size()),
                       p_old_values(quadrature_formula.size());
  std::vector< std::vector<double> >  s_values(model.n_phases()-1);
  for (auto & c: s_values)
    c.resize(face_quadrature_formula.size());
  // this one stores both saturation values and geomechanics
  std::vector<double>  extra_values(model.n_phases() - 1);

  const unsigned int q_point = 0;

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
      for (unsigned int c=0; c<model.n_phases() - 1; ++c)
      {
        fe_values.get_function_values(saturation[c], s_values[c]);
        extra_values[c] = s_values[c][q_point];
      }

      // std::cout << "cell: " << i << std::endl;
      const double pressure_value = p_values[q_point];
      const double pressure_value_old = p_old_values[q_point];

      cell_values.update(cell, pressure_value, extra_values);
      cell_values.update_wells(cell);

      // const double B_ii = cell_values.get_mass_matrix_entry();
      // double matrix_ii = B_ii/time_step + cell_values.get_J();
      // double rhs_i = B_ii/time_step*p_old + cell_values.get_Q();
      // double t_entry = 0;
      // new API
      double matrix_ii = cell_values.get_matrix_cell_entry(time_step);
      double rhs_i = cell_values.get_rhs_cell_entry(time_step,
                                                    pressure_value_old);
      // for debugging only
      // double face_entry = 0;

      cell->get_dof_indices(dof_indices);
      const unsigned int i = dof_indices[q_point];
      // std::cout << "ind " << i
      //           << "\tcell" <<cell->center()
      //           << "\tBii/time_step = " << B_ii/time_step
                // << "\t p_old = 0: " << p_old == 0.0)
                // << "\tQ = " << cell_values.get_Q()
                // << "\tJ = " << cell_values.get_J()
                // << std::endl;
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
            for (unsigned int c=0; c<model.n_phases() - 1; ++c)
            {
              fe_values.get_function_values(saturation[c], s_values[c]);
              extra_values[c] = s_values[c][q_point];
            }
            const double p_neighbor = p_values[q_point];

            normal = fe_face_values.normal_vector(q_point);
            const double dS = cell->face(f)->measure();  // face area

            // assemble local matrix and distribute
            neighbor_values.update(neighbor, p_neighbor, extra_values);
            cell_values.update_face_values(neighbor_values, normal, dS);

            // distribute
            neighbor->get_dof_indices(dof_indices_neighbor);
            const unsigned int j = dof_indices_neighbor[q_point];
            // const double T_face = cell_values.get_T_face();
            // matrix_ii += T_face;
            // t_entry += T_face;
            // rhs_i += cell_values.get_G_face();
            // system_matrix.add(i, j, -T_face);
            const double face_entry = cell_values.get_matrix_face_entry();
            matrix_ii += face_entry;
            rhs_i += cell_values.get_rhs_face_entry();
            system_matrix.add(i, j, -face_entry);
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
              fe_subface_values.reinit(cell, f, subface);
              // fe_face_values.reinit(cell, f);

              fe_values.get_function_values(relevant_solution, p_values);
              for (unsigned int c=0; c<model.n_phases() - 1; ++c)
              {
                fe_values.get_function_values(saturation[c], s_values[c]);
                extra_values[c] = s_values[c][0];
              }

              const double p_neighbor = p_values[q_point];
              normal = fe_subface_values.normal_vector(q_point);
              const double dS = fe_subface_values.JxW(q_point);

              // update neighbor
              neighbor_values.update(neighbor, p_neighbor, extra_values);
              // update face values
              cell_values.update_face_values(neighbor_values, normal, dS);

              // distribute
              neighbor->get_dof_indices(dof_indices_neighbor);
              const unsigned int j = dof_indices_neighbor[q_point];
              // const double T_face = cell_values.get_T_face();
              // matrix_ii += T_face;
              // t_entry += T_face;
              // rhs_i += cell_values.get_G_face();
              // system_matrix.add(i, j, -T_face);
              const double face_entry = cell_values.get_matrix_face_entry();
              matrix_ii += face_entry;
              rhs_i += cell_values.get_rhs_face_entry();
              system_matrix.add(i, j, -face_entry);
            }
          } // end case neighbor is finer

        } // end if face not at boundary
      }  // end face loop

      system_matrix.add(i, i, matrix_ii);
      rhs_vector[i] += rhs_i;

      // pcout << "i = " << i << std::endl;
      // if (i == 0)
      // {
      //   pcout << "\nPRESSURE"<< std::endl;
      //   pcout << "A(0, 0) = "<< system_matrix(i, i) << std::endl;
      //   pcout << "B(0, 0) = "<< B_ii/time_step << std::endl;
      //   pcout << "T(0, 0) = "<< t_entry << std::endl;
      //   pcout << "Q(0) = "<< cell_values.get_Q() << std::endl;
      //   pcout << "rhs(0) = "<< rhs_i << std::endl;
      //   pcout << std::endl;
      // }
      // if (i == 1)
      // {
      //   pcout << "\nPRESSURE"<< std::endl;
      //   pcout << "A(1, 1) = "<< system_matrix(i, i) << std::endl;
      //   pcout << "B(1, 1) = "<< B_ii/time_step << std::endl;
      //   pcout << "T(1, 1) = "<< t_entry << std::endl;
      //   pcout << "Q(1) = "<< cell_values.get_Q() << std::endl;
      //   pcout << "rhs(1) = "<< rhs_i << std::endl;
      //   pcout << std::endl;
      // }
      // if (i == 101)
      // {
      //   pcout << "\nPRESSURE"<< std::endl;
      //   pcout << "A(101, 101) = "<< system_matrix(i, i) << std::endl;
      //   pcout << "B(101, 101) = "<< B_ii/time_step << std::endl;
      //   pcout << "T(101, 101) = "<< t_entry << std::endl;
      //   pcout << "Q(101) = "<< cell_values.get_Q() << std::endl;
      //   pcout << "rhs(101) = "<< rhs_i << std::endl;
      //   pcout << std::endl;
      // }

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

  { // iterative solver
    TrilinosWrappers::SolverCG::AdditionalData additional_data_cg;
    TrilinosWrappers::SolverCG
        solver(solver_control, additional_data_cg);
    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data_amg;
    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix, additional_data_amg);
    solver.solve(system_matrix, solution, rhs_vector, preconditioner);
  }

  // { // direct solver
  //   TrilinosWrappers::SolverDirect
  //       solver(solver_control, TrilinosWrappers::SolverDirect::AdditionalData());
  //   solver.solve(system_matrix, solution, rhs_vector);
  // }
  return solver_control.last_step();
} // eom




template<int dim>
void
PressureSolver<dim>::
set_coupling(const DoFHandler<dim>               & solid_dof_handler,
             const TrilinosWrappers::MPI::Vector & displacement_vector)
{
  p_solid_dof_handler = & solid_dof_handler;
  p_displacement_vector = & p_displacement_vector;
  coupled_with_solid = true;
}  // eom



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

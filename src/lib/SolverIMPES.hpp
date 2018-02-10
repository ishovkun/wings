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
// #include <CellValues/CellValuesSaturation.hpp>
#include <ExtraFEData.hpp>



namespace FluidSolvers
{
using namespace dealii;


template <int dim>
class SolverIMPES
{
 public:
  SolverIMPES(MPI_Comm                                  &mpi_communicator_,
              parallel::distributed::Triangulation<dim> &triangulation_,
              const Model::Model<dim>                   &model_,
              ConditionalOStream                        &pcout_);
  ~SolverIMPES();
  /* setup degrees of freedom for the current triangulation
   * and allocate memory for solution vectors */
  void setup_dofs();
  /* generic method to assemble finite volume system
   * it is used in the assembly of pressure system and explicit saturation
   * solution */
  void
  assemble_finite_volume_system(CellValues::CellValuesBase<dim> & cell_values,
                                CellValues::CellValuesBase<dim> & neighbor_values,
                                const double                      time_step,
                                const bool                        assemble_matrix,
                                TrilinosWrappers::MPI::Vector   & rhs_vector,
                                const int                         phase);
  // Implicit pressure system: Fill matrix and rhs vector
  void assemble_pressure_system(CellValues::CellValuesBase<dim> & cell_values,
                                CellValues::CellValuesBase<dim> & neighbor_values,
                                const double                      time_step);
  /*
   * solve saturation system explicitly. This method makes use of
   * assemble_finite_volume_system.
   */
  void solve_saturation_system(CellValues::CellValuesBase<dim>       & cell_values,
                               CellValues::CellValuesBase<dim>       & neighbor_values,
                               const double                            time_step,
                               const int                               phase);
  /*
   * solve linear system syste_matrix*pressure_solution = rhs_vector
   * returns the number of solver steps
   */
  unsigned int solve_pressure_system();
  // accessing private members
  const TrilinosWrappers::SparseMatrix & get_system_matrix();
  const TrilinosWrappers::MPI::Vector  & get_rhs_vector();
  const DoFHandler<dim>                & get_dof_handler();
  const FE_DGQ<dim>                    & get_fe();

 private:
  MPI_Comm                                  & mpi_communicator;
  parallel::distributed::Triangulation<dim> & triangulation;
  DoFHandler<dim>                             dof_handler;
  FE_DGQ<dim>                                 fe;
  const Model::Model<dim>                   & model;
  ConditionalOStream                        & pcout;

  // Matrices and vectors
  TrilinosWrappers::SparseMatrix             system_matrix;
  TrilinosWrappers::MPI::Vector              rhs_vector;

 public:
  TrilinosWrappers::MPI::Vector              pressure_solution,
                                             pressure_relevant_solution,
                                             pressure_old_solution;
  std::vector<TrilinosWrappers::MPI::Vector> saturation_solution,
                                             saturation_relevant_solution,
                                             saturation_old_solution;
  // partitioning
  IndexSet                      locally_owned_dofs, locally_relevant_dofs;
};


template <int dim>
SolverIMPES<dim>::
SolverIMPES(MPI_Comm                                  &mpi_communicator_,
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
    saturation_solution(model.n_phases()),
    saturation_relevant_solution(model.n_phases()),
    saturation_old_solution(model.n_phases() - 1)  // old solution n-1 phases
{}  // eom


template <int dim>
SolverIMPES<dim>::~SolverIMPES()
{
  dof_handler.clear();
}  // eom


template <int dim>
void SolverIMPES<dim>::setup_dofs()
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
    pressure_solution.reinit(locally_owned_dofs, mpi_communicator);
    pressure_old_solution.reinit(locally_relevant_dofs, mpi_communicator);
    pressure_relevant_solution.reinit(locally_relevant_dofs, mpi_communicator);
    rhs_vector.reinit(locally_owned_dofs, locally_relevant_dofs,
                      mpi_communicator, /* omit-zeros= */ true);
    for (unsigned int p=0; p<model.n_phases(); ++p)
    {
      saturation_solution[p].reinit(locally_owned_dofs, mpi_communicator);
      saturation_relevant_solution[p].reinit(locally_relevant_dofs, mpi_communicator);
      // old solution stores only n-1 phases
      if (p < model.n_phases() - 1)
        saturation_old_solution[p].reinit(locally_relevant_dofs, mpi_communicator);
    }
  }  // end setup vectors
} // eom



template <int dim>
void
SolverIMPES<dim>::
assemble_finite_volume_system(CellValues::CellValuesBase<dim> & cell_values,
                              CellValues::CellValuesBase<dim> & neighbor_values,
                              const double                      time_step,
                              const bool                        assemble_matrix,
                              TrilinosWrappers::MPI::Vector   & rhs_vector,
                              const int                         phase)
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
  Tensor<1, dim>                      normal;
  std::vector<double>                 p_values(quadrature_formula.size()),
                                      p_old_values(quadrature_formula.size());
  std::vector< std::vector<double> >  s_values(model.n_phases()-1);
  for (auto & c: s_values)
    c.resize(face_quadrature_formula.size());
  // this one stores both saturation values and geomechanics
  std::vector<double>                 extra_values(model.n_phases() - 1);

  const unsigned int q_point = 0;

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

  if (assemble_matrix)
    system_matrix = 0;
  rhs_vector = 0;

  for (; cell!=endc; ++cell)
  {
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(pressure_old_solution, p_old_values);
      fe_values.get_function_values(pressure_relevant_solution, p_values);
      for (unsigned int c=0; c<model.n_phases() - 1; ++c)
      {
        fe_values.get_function_values(saturation_relevant_solution[c],
                                      s_values[c]);
        extra_values[c] = s_values[c][q_point];
      }

      // std::cout << "cell: " << i << std::endl;
      const double pressure_value = p_values[q_point];
      const double pressure_value_old = p_old_values[q_point];

      cell_values.update(cell, pressure_value, extra_values);
      cell_values.update_wells(cell);

      double rhs_i = cell_values.get_rhs_cell_entry(time_step,
                                                    pressure_value,
                                                    pressure_value_old,
                                                    phase);
      double matrix_ii = 0;
      if (assemble_matrix)
        matrix_ii = cell_values.get_matrix_cell_entry(time_step);

      cell->get_dof_indices(dof_indices);
      const unsigned int i = dof_indices[q_point];

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

            fe_values.get_function_values(pressure_relevant_solution, p_values);
            for (unsigned int c=0; c<model.n_phases() - 1; ++c)
            {
              fe_values.get_function_values(saturation_relevant_solution[c],
                                            s_values[c]);
              extra_values[c] = s_values[c][q_point];
            }
            const double p_neighbor = p_values[q_point];

            normal = fe_face_values.normal_vector(q_point);
            const double dS = cell->face(f)->measure();  // face area

            // assemble local matrix and distribute
            neighbor_values.update(neighbor, p_neighbor, extra_values);
            cell_values.update_face_values(neighbor_values, normal, dS);

            // distribute
            rhs_i += cell_values.get_rhs_face_entry(time_step, phase);
            if (assemble_matrix)
            {
              neighbor->get_dof_indices(dof_indices_neighbor);
              const unsigned int j = dof_indices_neighbor[q_point];
              const double face_entry = cell_values.get_matrix_face_entry();
              matrix_ii += face_entry;
              system_matrix.add(i, j, -face_entry);
            }
          } // end case neighbor coarser or same
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

              fe_values.get_function_values(pressure_relevant_solution, p_values);
              for (unsigned int c=0; c<model.n_phases() - 1; ++c)
              {
                fe_values.get_function_values(saturation_relevant_solution[c],
                                              s_values[c]);
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
              rhs_i += cell_values.get_rhs_face_entry(time_step, phase);
              if (assemble_matrix)
              {
                const double face_entry = cell_values.get_matrix_face_entry();
                matrix_ii += face_entry;
                neighbor->get_dof_indices(dof_indices_neighbor);
                const unsigned int j = dof_indices_neighbor[q_point];
                system_matrix.add(i, j, -face_entry);
              }
            } // end subface loop
          } // end case neighbor is finer

        } // end if face not at boundary
      }  // end face loop

      if (assemble_matrix)
        system_matrix.add(i, i, matrix_ii);
      rhs_vector[i] += rhs_i;

    } // end local cells
  } // end cells loop

  if (assemble_matrix)
    system_matrix.compress(VectorOperation::add);
  rhs_vector.compress(VectorOperation::add);
} // eom



template <int dim>
void
SolverIMPES<dim>::
assemble_pressure_system(CellValues::CellValuesBase<dim> & cell_values,
                         CellValues::CellValuesBase<dim> & neighbor_values,
                         const double                      time_step)
{
  assemble_finite_volume_system(cell_values, neighbor_values,
                                time_step, /* assemble_matrix = */ true,
                                rhs_vector, /* phase = */ 0);
} // eom



template <int dim>
void
SolverIMPES<dim>::
solve_saturation_system(CellValues::CellValuesBase<dim> & cell_values,
                        CellValues::CellValuesBase<dim> & neighbor_values,
                        const double                      time_step,
                        const int                         phase)
{
  assemble_finite_volume_system(cell_values, neighbor_values,
                                time_step, /* assemble_matrix = */ false,
                                rhs_vector, phase);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  const int q_point = 0;

  std::pair<double,double> saturation_limits =
      model.get_saturation_limits(phase);

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned)
    {
      cell->get_dof_indices(dof_indices);
      const unsigned int i = dof_indices[q_point];
      const double old_saturation = saturation_relevant_solution[phase][i];
      double increment = rhs_vector[i];
      if (old_saturation + increment > saturation_limits.second)
        increment = saturation_limits.second - old_saturation;
      else if (old_saturation + increment < saturation_limits.first)
        increment = saturation_limits.first - old_saturation;

      saturation_relevant_solution[phase][i] =
          old_saturation + increment;
    }

  saturation_solution[phase].compress(VectorOperation::insert);
}  // eom



template <int dim>
unsigned int
SolverIMPES<dim>::solve_pressure_system()
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
    // solver.solve(system_matrix, solution, rhs_vector, preconditioner);
    solver.solve(system_matrix, pressure_solution, rhs_vector, preconditioner);
  }

  // { // direct solver
  //   TrilinosWrappers::SolverDirect
  //       solver(solver_control, TrilinosWrappers::SolverDirect::AdditionalData());
  //   solver.solve(system_matrix, solution, rhs_vector);
  // }

  return solver_control.last_step();
} // eom



// template <int dim>
// void
// SolverIMPES<dim>::
// solve_saturation_system(CellValues::CellValuesBase<dim> & cell_values,
//                         CellValues::CellValuesBase<dim> & neighbor_values,
//                         const double                      time_step)
// {
//   // Only one integration point in FVM
//   QGauss<dim>   quadrature_formula(1);
//   QGauss<dim-1> face_quadrature_formula(1);

//   FEValues<dim> fe_values(fe, quadrature_formula, update_values);
//   FEValues<dim> fe_values_neighbor(fe, quadrature_formula, update_values);

//   // the following two objects only get geometry data
//   FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
//                                    update_normal_vectors);
//   // We need JxW flag for subfaces since there is no
//   // method to determine sub face area in triangulation class
//   FESubfaceValues<dim> fe_subface_values(fe, face_quadrature_formula,
//                                          update_normal_vectors |
//                                          update_JxW_values);

//   const unsigned int dofs_per_cell = fe.dofs_per_cell;
//   std::vector<types::global_dof_index>
//       dof_indices(dofs_per_cell),
//       dof_indices_neighbor(dofs_per_cell);

//   // objects to store local data
//   Tensor<1, dim>       normal;
//   std::vector<double>  p_values(quadrature_formula.size()),
//                        p_old_values(quadrature_formula.size());

//   std::vector< std::vector<double> >  s_values(model.n_phases()-1);
//   for (auto & c: s_values)
//     c.resize(face_quadrature_formula.size());

//   std::vector<double>  extra_values(model.n_phases() - 1);

//   const double So_rw = model.residual_saturation_oil();
//   const double Sw_crit = model.residual_saturation_water();
//   const unsigned int q_point = 0;

//   typename DoFHandler<dim>::active_cell_iterator
//       cell = dof_handler.begin_active(),
//       endc = dof_handler.end();

//   for (; cell!=endc; ++cell)
//     if (cell->is_locally_owned())
//     {
//       fe_values.reinit(cell);
//       fe_values.get_function_values(old_pressure_solution, p_old_values);
//       fe_values.get_function_values(pressure_solution, p_values);
//       for (unsigned int c=0; c<model.n_phases() - 1; ++c)
//       {
//         fe_values.get_function_values(relevant_solution[c], s_values[c]);
//         extra_values[c] = s_values[c][0];
//       }

//       const double p_old = p_old_values[q_point];
//       const double p = p_values[q_point];
//       const double Sw_old = extra_values[q_point];

//       // std::cout << "value1 c1w = " << cell_values.c1p << std::endl;
//       // std::cout << "value1 c1p = " << cell_values.c1w << std::endl;
//       cell_values.update(cell, p, extra_values);
//       // std::cout << "value2 c1p = " << cell_values.c1p << std::endl;
//       // std::cout << "value2 c1w = " << cell_values.c1w << std::endl;
//       // std::cout << "value2 c1p/c1w = " << cell_values.c1p / cell_values.c1w << std::endl;
//       // std::cout << "value_weird c1p/c1w = " << cell_values.get_B(0) << std::endl;
//       cell_values.update_wells(cell, p);

//       double solution_increment =
//           cell_values.get_rhs_cell_entry(time_step, p, p_old, 0);

//       cell->get_dof_indices(dof_indices);
//       const unsigned int i = dof_indices[q_point];

//       for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//       {
//         if (cell->at_boundary(f) == false)
//         {
//           if((cell->neighbor(f)->level() == cell->level() &&
//               cell->neighbor(f)->has_children() == false) ||
//              cell->neighbor_is_coarser(f))
//           {
//             const auto & neighbor = cell->neighbor(f);
//             fe_values.reinit(neighbor);
//             fe_face_values.reinit(cell, f);

//             fe_values.get_function_values(pressure_solution, p_values);
//             for (unsigned int c=0; c<model.n_phases() - 1; ++c)
//             {
//               fe_values.get_function_values(relevant_solution[c], s_values[c]);
//               extra_values[c] = s_values[c][q_point];
//             }
//             const double p_neighbor = p_values[q_point];

//             normal = fe_face_values.normal_vector(q_point);
//             const double dS = cell->face(f)->measure();  // face area

//             // assemble local matrix and distribute
//             neighbor_values.update(neighbor, p_neighbor, extra_values);
//             cell_values.update_face_values(neighbor_values, normal, dS);

//             solution_increment += cell_values.get_rhs_face_entry(time_step, 0);
//           }
//           else if ((cell->neighbor(f)->level() == cell->level()) &&
//                    (cell->neighbor(f)->has_children() == true))
//           {
//             for (unsigned int subface=0;
//                  subface<cell->face(f)->n_children(); ++subface)
//             {
//               // compute parameters
//               const auto & neighbor
//                   = cell->neighbor_child_on_subface(f, subface);

//               fe_values.reinit(neighbor);
//               fe_subface_values.reinit(cell, f, subface);

//               fe_values.get_function_values(pressure_solution, p_values);
//               const double p_neighbor = p_values[q_point];
//               for (unsigned int c=0; c<model.n_phases()-1; ++c)
//                 fe_values.get_function_values(relevant_solution[c], s_values[c]);

//               normal = fe_subface_values.normal_vector(q_point); // 0 is gauss point
//               const double dS = fe_subface_values.JxW(q_point);

//               // update neighbor
//               neighbor_values.update(neighbor, p_neighbor, extra_values);

//               // update face values
//               cell_values.update_face_values(neighbor_values, normal, dS);

//               // distribute
//               solution_increment += cell_values.get_rhs_face_entry(time_step, 0);
//               // solution_increment +=
//               //     time_step *
//               //     (
//               //         cell_values.get_T_face(0)
//               //         -
//               //         cell_values.get_G_face(0)
//               //      )
//               //     ;
//             }
//           } // end case neighbor is finer

//         } // end if face not at boundary
//       }  // end face loop

//       // assert that we are in bounds
//       if (Sw_old + solution_increment > (1.0 - So_rw))
//         solution_increment = (1.0 - So_rw) - Sw_old;
//       else if (Sw_old + solution_increment < Sw_crit)
//         solution_increment = Sw_crit - Sw_old;

//       // if (i == 0)
//       // {
//       //   pcout << "\nSATURATION"<< std::endl;
//       //   pcout << "dp = "<< p - p_old << std::endl;
//       //   pcout << "dSw = " << solution_increment << std::endl;
//       //   pcout << "Qw = " << time_step*cell_values.get_Q(0) << std::endl;
//       //   pcout << "Tw = " << cell_values.get_T_face(0) << std::endl;
//       //   pcout << "C1P/C1W = " << cell_values.get_B(0)
//       //         << std::endl;

//       //   pcout << std::endl;
//       // }
//       solution[0][i] = Sw_old + solution_increment;
//       solution[1][i] = 1.0 - (Sw_old + solution_increment);
//     } // end cells loop

//   // solution[0].compress(VectorOperation::add);
//   // solution[1].compress(VectorOperation::add);
//   solution[0].compress(VectorOperation::insert);
//   solution[1].compress(VectorOperation::insert);
// }  // eom



template <int dim>
const TrilinosWrappers::SparseMatrix&
SolverIMPES<dim>::get_system_matrix()
{
  return system_matrix;
}  // eom



template <int dim>
const TrilinosWrappers::MPI::Vector&
SolverIMPES<dim>::get_rhs_vector()
{
  return rhs_vector;
}  // eom



template <int dim>
const DoFHandler<dim> &
SolverIMPES<dim>::get_dof_handler()
{
  return dof_handler;
}  // eom


template <int dim>
const FE_DGQ<dim> &
SolverIMPES<dim>::get_fe()
{
  return fe;
}  // eom
}  // end of namespace

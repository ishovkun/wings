#pragma once

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <CellValues/CellValuesSaturation.hpp>


namespace FluidSolvers
{
using namespace dealii;


template <int dim>
class SaturationSolver
{
 public:
  SaturationSolver(MPI_Comm                 &mpi_communicator_,
                   const DoFHandler<dim>    &dof_handler_,
                   const Model::Model<dim>  &model_,
                   ConditionalOStream       &pcout_);
  void setup_dofs(IndexSet &locally_owned_dofs,
                  IndexSet &locally_relevant_dofs);
  void
  solve(CellValues::CellValuesSaturation<dim> &cell_values,
        CellValues::CellValuesBase<dim>       &neighbor_values,
        const double                           time_step,
        const TrilinosWrappers::MPI::Vector   &pressure_solution,
        const TrilinosWrappers::MPI::Vector   &old_pressure_solution);

  const unsigned int                        n_phases;
 private:
  MPI_Comm                                  &mpi_communicator;
  const DoFHandler<dim>                     &dof_handler;
  const Model::Model<dim>                   &model;
  ConditionalOStream                        &pcout;
 public:
  std::vector<TrilinosWrappers::MPI::Vector>
  solution, relevant_solution, old_solution;
  TrilinosWrappers::MPI::Vector rhs_vector;
};


template <int dim>
SaturationSolver<dim>::
SaturationSolver(MPI_Comm                   &mpi_communicator_,
                 const DoFHandler<dim>      &dof_handler_,
                 const Model::Model<dim>    &model_,
                 ConditionalOStream         &pcout_)
    :
    n_phases(model_.n_phases()),
    mpi_communicator(mpi_communicator_),
    dof_handler(dof_handler_),
    model(model_),
    pcout(pcout_)
{}


template <int dim>
void
SaturationSolver<dim>::setup_dofs(IndexSet &locally_owned_dofs,
                                  IndexSet &locally_relevant_dofs)
{
  if (solution.size() != n_phases)
  {
    solution.resize(n_phases);
    relevant_solution.resize(n_phases);
    old_solution.resize(n_phases);
  }

  for (unsigned int p=0; p<n_phases; ++p)
  {
    solution[p].reinit(locally_owned_dofs, mpi_communicator);
    relevant_solution[p].reinit(locally_relevant_dofs, mpi_communicator);
    old_solution[p].reinit(locally_relevant_dofs, mpi_communicator);
  }

  rhs_vector.reinit(locally_owned_dofs, locally_relevant_dofs,
                    mpi_communicator, /* omit-zeros=*/ true);
}  // eom



template <int dim>
void
SaturationSolver<dim>::
solve(CellValues::CellValuesSaturation<dim> &cell_values,
      CellValues::CellValuesBase<dim>       &neighbor_values,
      const double                           time_step,
      const TrilinosWrappers::MPI::Vector   &pressure_solution,
      const TrilinosWrappers::MPI::Vector   &old_pressure_solution)
{
  // Only one integration point in FVM
  QGauss<dim>       quadrature_formula(1);
  QGauss<dim-1>     face_quadrature_formula(1);

  const auto & fe = dof_handler.get_fe();
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

  std::vector<double>  extra_values(model.n_phases() - 1);

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(old_pressure_solution, p_old_values);
      fe_values.get_function_values(pressure_solution, p_values);

      const double p_old = p_old_values[0];
      const double p = p_values[0];

      for (unsigned int c=0; c<model.n_phases() - 1; ++c)
      {
        fe_values.get_function_values(relevant_solution[c], s_values[c]);
        extra_values[c] = s_values[c][0];
      }

      cell_values.update(cell, p, extra_values);
      cell_values.update_wells(cell, p);

      // const double B_ii = cell_values.get_mass_matrix_entry();
      // double matrix_ii = B_ii/time_step + cell_values.get_J();
      // double rhs_i = B_ii/time_step*p_old + cell_values.get_Q();
      double solution_increment =
          time_step *
          (
              -
              cell_values.get_B(0) * (p - p_old)/ time_step
              // -
              // cell_values.get_E() * ( div_e - div_e_old ) / time_step
              +
              cell_values.get_Q(0)
           )
          ;
      // pcout << "cell " << cell->center() << std::endl;
      // pcout <<  "dp entry" << cell_values.get_B(0) * (p - p_old) << std::endl;
      // pcout << "Q " << time_step*cell_values.get_Q(0) << std::endl;


      cell->get_dof_indices(dof_indices);
      const unsigned int i = dof_indices[0];
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

            fe_values.get_function_values(pressure_solution, p_values);
            const double p_neighbor = p_values[0];
            for (unsigned int c=0; c<model.n_phases() - 1; ++c)
            {
              fe_values.get_function_values(relevant_solution[c], s_values[c]);
              extra_values[c] = s_values[c][0];
            }

            normal = fe_face_values.normal_vector(0); // 0 is gauss point
            const double dS = cell->face(f)->measure();  // face area

            // assemble local matrix and distribute
            neighbor_values.update(neighbor, p_neighbor, extra_values);
            cell_values.update_face_values(neighbor_values, normal, dS);

            solution_increment +=
            time_step *
                (
                    cell_values.get_T_face(0)
                    -
                    cell_values.get_G_face(0)
                 )
                ;

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

              fe_values.get_function_values(pressure_solution, p_values);
              const double p_neighbor = p_values[0];
              for (unsigned int c=0; c<model.n_phases() - 1; ++c)
                fe_values.get_function_values(relevant_solution[c], s_values[c]);

              fe_subface_values.reinit(cell, f, subface);
              normal = fe_subface_values.normal_vector(0); // 0 is gauss point
              const double dS = fe_subface_values.JxW(0);

              // update neighbor
              neighbor_values.update(neighbor, p_neighbor, extra_values);

              // update face values
              cell_values.update_face_values(neighbor_values, normal, dS);

              // distribute
              solution_increment +=
                  time_step *
                  (
                      cell_values.get_T_face(0)
                      -
                      cell_values.get_G_face(0)
                   )
                  ;
            }
          } // end case neighbor is finer

        } // end if face not at boundary
      }  // end face loop

      solution[0][i] += solution_increment;
      solution[1][i] = 1.0 - solution[0][i];
    } // end cells loop

  solution[0].compress(VectorOperation::add);
  solution[1].compress(VectorOperation::add);
}


} // end of namespace

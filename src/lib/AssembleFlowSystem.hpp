#pragma once

// dealii modules
#include <deal.II/dofs/dof_handler.h>
// #include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

// wings modules
// #include <CellValues/CellValuesBase.hpp>
#include <CellValues/CellValuesPressure.hpp>

namespace FluidSolvers
{
using namespace dealii;


template <int dim, typename VectorType, typename MatrixType>
void assemble_flow_system(const DoFHandler<dim>               & fluid_dof_handler,
                          const DoFHandler<dim>               & solid_dof_handler,
                          const VectorType                    & pressure,
                          const VectorType                    & old_pressure,
                          const std::vector<VectorType>       & saturations,
                          const VectorType                    & solid_solution,
                          const VectorType                    & old_solid_solution,
                          const FEValuesExtractors::Vector    & displacement_extractor,
                          // CellValues::CellValuesBase<dim>  & cell_values,
                          // CellValues::CellValuesBase<dim>  & neighbor_values,
                          CellValues::CellValuesPressure<dim> & cell_values,
                          CellValues::CellValuesPressure<dim> & neighbor_values,
                          MatrixType                          & system_matrix,
                          VectorType                          & rhs_vector,
                          const double                        time_step,
                          const unsigned int                  n_phases,
                          const bool                          coupled_with_solid,
                          const bool                          assemble_matrix)
{
  // Only one integration point in FVM
  QGauss<dim>   quadrature_formula(1);
  QGauss<dim-1> face_quadrature_formula(1);

  const auto & fe_fluid = fluid_dof_handler.get_fe();

  FEValues<dim> fe_values(fe_fluid, quadrature_formula, update_values);
  FEValues<dim> fe_values_neighbor(fe_fluid, quadrature_formula, update_values);
  // the following two objects only get geometry data
  FEFaceValues<dim> fe_face_values(fe_fluid, face_quadrature_formula,
                                   update_normal_vectors);
  // We need JxW flag for subfaces since there is no
  // method to determine sub face area in triangulation class
  FESubfaceValues<dim> fe_subface_values(fe_fluid, face_quadrature_formula,
                                         update_normal_vectors |
                                         update_JxW_values);
  FEValues<dim> * p_fe_values_solid = NULL;
  if (coupled_with_solid)
    p_fe_values_solid = new FEValues<dim>(solid_dof_handler.get_fe(),
                                          quadrature_formula, update_gradients);
  auto & fe_values_solid = * p_fe_values_solid;

  const unsigned int dofs_per_cell = fe_fluid.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();
  const unsigned int q_point       = 0;

  std::vector<types::global_dof_index>
      dof_indices(dofs_per_cell),
      dof_indices_neighbor(dofs_per_cell);

  // objects to store local data
  Tensor<1, dim>                      face_normal;
  std::vector<double>                 p_values(n_q_points);
  std::vector<double>                 p_old_values(n_q_points);
  std::vector<double>                 div_u_values(n_q_points);
  std::vector<double>                 div_old_u_values(n_q_points);
  std::vector< std::vector<double> >  s_values(n_phases);
  for (auto & c: s_values)
    c.resize(face_quadrature_formula.size());

  CellValues::SolutionValues solution_values;
  CellValues::FaceGeometry   face_values;

  typename DoFHandler<dim>::active_cell_iterator
      cell = fluid_dof_handler.begin_active(),
      // trick to place solid_cell in cell loop condition
      solid_cell = fluid_dof_handler.begin_active(),
      endc = fluid_dof_handler.end();
  if (coupled_with_solid)
    solid_cell = solid_dof_handler.begin_active();

  if (assemble_matrix)
    system_matrix = 0;
  rhs_vector = 0;

  for (; cell!=endc; ++cell, ++solid_cell)
  {
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(pressure, p_values);
      fe_values.get_function_values(old_pressure, p_old_values);
      for (unsigned int c=0; c<n_phases - 1; ++c)
      {
        fe_values.get_function_values(saturations[c], s_values[c]);
        solution_values.saturation[c] = s_values[c][q_point];
      }
      if (coupled_with_solid)
      {
        fe_values_solid.reinit(solid_cell);
        fe_values_solid[displacement_extractor].
            get_function_divergences(solid_solution, div_u_values);
        fe_values_solid[displacement_extractor].
            get_function_divergences(old_solid_solution, div_old_u_values);
        solution_values.div_u = div_u_values[q_point];
        solution_values.div_old_u = div_old_u_values[q_point];
      }

      const double pressure_value = p_values[q_point];
      const double pressure_value_old = p_old_values[q_point];
      solution_values.pressure = pressure_value;

      cell_values.update(cell, solution_values);
      cell_values.update_wells(cell);

      double matrix_ii = cell_values.get_matrix_cell_entry(time_step);
      double rhs_i = cell_values.get_rhs_cell_entry(time_step,
                                                    pressure_value,
                                                    pressure_value_old);

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

            fe_values.get_function_values(pressure, p_values);
            for (unsigned int c=0; c<n_phases - 1; ++c)
            {
              fe_values.get_function_values(saturations[c], s_values[c]);
              solution_values.saturation[c] = s_values[c][q_point];
            }
            if (coupled_with_solid)
            {
              fe_values_solid.reinit(solid_cell->neighbor(f));
              fe_values_solid[displacement_extractor].
                  get_function_divergences(solid_solution, div_u_values);
              fe_values_solid[displacement_extractor].
                  get_function_divergences(old_solid_solution, div_old_u_values);
              solution_values.div_u = div_u_values[q_point];
              solution_values.div_old_u = div_old_u_values[q_point];
            }
            solution_values.pressure = p_values[q_point];
            face_values.normal = fe_face_values.normal_vector(q_point);
            face_values.area = cell->face(f)->measure();

            // assemble local matrix and distribute
            neighbor_values.update(neighbor, solution_values);
            cell_values.update_face_values(neighbor_values, face_values);

            // distribute
            neighbor->get_dof_indices(dof_indices_neighbor);
            const unsigned int j = dof_indices_neighbor[q_point];
            const double face_entry = cell_values.get_matrix_face_entry();
            matrix_ii += face_entry;
            rhs_i += cell_values.get_rhs_face_entry(time_step);
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

              fe_values.get_function_values(pressure, p_values);
              for (unsigned int c=0; c<n_phases - 1; ++c)
              {
                fe_values.get_function_values(saturations[c], s_values[c]);
                solution_values.saturation[c] = s_values[c][q_point];
              }
              if (coupled_with_solid)
              {
                const auto & solid_neighbor =
                    solid_cell->neighbor_child_on_subface(f, subface);
                fe_values_solid.reinit(solid_neighbor);
                fe_values_solid[displacement_extractor].
                    get_function_divergences(solid_solution, div_u_values);
                fe_values_solid[displacement_extractor].
                    get_function_divergences(old_solid_solution, div_old_u_values);
                solution_values.div_u = div_u_values[q_point];
                solution_values.div_old_u = div_old_u_values[q_point];
              }

              solution_values.pressure = p_values[q_point];
              face_values.normal = fe_subface_values.normal_vector(q_point);
              face_values.area = fe_subface_values.JxW(q_point);

              // update neighbor
              neighbor_values.update(neighbor, solution_values);
              // update face values
              cell_values.update_face_values(neighbor_values, face_values);

              // distribute
              neighbor->get_dof_indices(dof_indices_neighbor);
              const unsigned int j = dof_indices_neighbor[q_point];
              const double face_entry = cell_values.get_matrix_face_entry();
              matrix_ii += face_entry;
              rhs_i += cell_values.get_rhs_face_entry(time_step);
              system_matrix.add(i, j, -face_entry);
            }
          } // end case neighbor is finer

        } // end if face not at boundary
      }  // end face loop

      if (assemble_matrix)
        system_matrix.add(i, i, matrix_ii);
      rhs_vector[i] += rhs_i;
    } // end local cells
  } // end cell loop

  if (assemble_matrix)
    system_matrix.compress(VectorOperation::add);
  rhs_vector.compress(VectorOperation::add);
  delete p_fe_values_solid;
} // end of method

} // end of namespace


namespace FluidSolvers
{
using namespace dealii;


template <int dim>
void assemble_system_fvm
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
  FEValues<dim> * p_fe_values_solid = NULL;
  if (coupled_with_solid)
    p_fe_values_solid = new FEValues<dim>(p_solid_dof_handler->get_fe(),
                                          quadrature_formula, update_gradients);
  auto & fe_values_solid = * p_fe_values_solid;

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();
  const unsigned int q_point       = 0;

  std::vector<types::global_dof_index>
      dof_indices(dofs_per_cell),
      dof_indices_neighbor(dofs_per_cell);

  // objects to store local data
  Tensor<1, dim>       normal;
  std::vector<double>                 p_values(n_q_points);
  std::vector<double>                 p_old_values(n_q_points);
  std::vector<double>                 div_u_values(n_q_points);
  std::vector<double>                 div_old_u_values(n_q_points);
  std::vector< std::vector<double> >  s_values(model.n_phases());
  for (auto & c: s_values)
    c.resize(face_quadrature_formula.size());

  CellValues::ExtraValues extra_values;
  CellValues::FaceValues  face_values;

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      // trick to place solid_cell in cell loop condition
      solid_cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  if (coupled_with_solid)
    solid_cell = p_solid_dof_handler->begin_active();

  system_matrix = 0;
  rhs_vector = 0;

  for (; cell!=endc; ++cell, ++solid_cell)
  {
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(pressure_old, p_old_values);
      fe_values.get_function_values(pressure_relevant, p_values);
      for (unsigned int c=0; c<model.n_phases() - 1; ++c)
      {
        fe_values.get_function_values(saturation_relevant[c], s_values[c]);
        extra_values.saturation[c] = s_values[c][q_point];
      }
      if (coupled_with_solid)
      {
        fe_values_solid.reinit(solid_cell);
        fe_values_solid[*p_displacement_extractor].
            get_function_divergences(*p_displacement, div_u_values);
        fe_values_solid[*p_displacement_extractor].
            get_function_divergences(*p_old_displacement, div_old_u_values);
        extra_values.div_u = div_u_values[q_point];
        extra_values.div_old_u = div_old_u_values[q_point];
      }

      const double pressure_value = p_values[q_point];
      const double pressure_value_old = p_old_values[q_point];
      extra_values.pressure = pressure_value;

      cell_values.update(cell, extra_values);
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

            fe_values.get_function_values(pressure_relevant, p_values);
            for (unsigned int c=0; c<model.n_phases() - 1; ++c)
            {
              fe_values.get_function_values(saturation_relevant[c], s_values[c]);
              extra_values.saturation[c] = s_values[c][q_point];
            }
            if (coupled_with_solid)
            {
              fe_values_solid.reinit(solid_cell->neighbor(f));
              fe_values_solid[*p_displacement_extractor].
                  get_function_divergences(*p_displacement, div_u_values);
              fe_values_solid[*p_displacement_extractor].
                  get_function_divergences(*p_old_displacement, div_old_u_values);
              extra_values.div_u = div_u_values[q_point];
              extra_values.div_old_u = div_old_u_values[q_point];
            }
            const double p_neighbor = p_values[q_point];

            facevalues.normal
            normal = fe_face_values.normal_vector(q_point);
            const double dS = cell->face(f)->measure();  // face area

            // assemble local matrix and distribute
            cell_values_neighbor.update(neighbor, p_neighbor, extra_values);
            cell_values.update_face_values(cell_values_neighbor, normal, dS);

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

              fe_values.get_function_values(pressure_relevant, p_values);
              for (unsigned int c=0; c<model.n_phases() - 1; ++c)
              {
                fe_values.get_function_values(saturation_relevant[c], s_values[c]);
                extra_values.saturation[c] = s_values[c][q_point];
              }
              if (coupled_with_solid)
              {
                const auto & solid_neighbor =
                    solid_cell->neighbor_child_on_subface(f, subface);
                fe_values_solid.reinit(solid_neighbor);
                fe_values_solid[*p_displacement_extractor].
                    get_function_divergences(*p_displacement, div_u_values);
                fe_values_solid[*p_displacement_extractor].
                    get_function_divergences(*p_old_displacement, div_old_u_values);
                extra_values.div_u = div_u_values[q_point];
                extra_values.div_old_u = div_old_u_values[q_point];
              }

              const double p_neighbor = p_values[q_point];
              normal = fe_subface_values.normal_vector(q_point);
              const double dS = fe_subface_values.JxW(q_point);

              // update neighbor
              cell_values_neighbor.update(neighbor, p_neighbor, extra_values);
              // update face values
              cell_values.update_face_values(cell_values_neighbor, normal, dS);

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

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
#include <deal.II/numerics/data_out.h>
// Trilinos stuff
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

// Custom modules
#include <FluidSolverBase.hpp>
#include <Equations/IMPESPressure.hpp>
#include <Equations/IMPESSaturation.hpp>
#include <AssembleFlowSystem.hpp>
#include <Model.hpp>
#include <ScaleOutputVector.hpp>
#include <FEFunction/FEFunctionPS.hpp>

namespace Wings {

namespace FluidSolvers
{
using namespace dealii;


template <int dim, int n_phases>
class SolverIMPES : public FluidSolverBase<dim,n_phases>
{
 public:
  /* TODO: initialization description */
  SolverIMPES(MPI_Comm                                  & mpi_communicator_,
              parallel::distributed::Triangulation<dim> & triangulation_,
              const Model::Model<dim>                   & model_,
              ConditionalOStream                        & pcout_,
              Equations::IMPESPressure<n_phases>        & cell_values,
              Equations::IMPESSaturation<n_phases>      & cell_values_saturation);
  ~SolverIMPES();
  /* setup degrees of freedom for the current triangulation
   * and allocate memory for solution vectors */
  void setup_dofs() override;
  // save solution for old iteration to compute difference later
  // virtual void save_solution() override;
  // Implicit pressure system: Fill matrix and rhs vector
  void assemble_pressure_system(const double time_step);
  /*
   * solve saturation system explicitly.
   */
  void solve_saturation_system(const double time_step);
  /*
   * solve linear system syste_matrix*pressure_solution = rhs_vector
   * returns the number of solver steps
   */
  unsigned int solve_pressure_system();
  /*
   * Solve pressure system, and then explicitly solve
   * for saturations
   */
  unsigned int solve_time_step(const double time_step) override;
  // give solver access to solid dofs and solution vector
  void set_coupling(const DoFHandler<dim>               & solid_dof_handler,
                    const TrilinosWrappers::MPI::Vector & displacement_vector,
                    const TrilinosWrappers::MPI::Vector & old_displacement_vector,
                    const FEValuesExtractors::Vector    & extractor);
  /*
   * Attach pressure and saturation vectors to the DataOut object.
   * This method is used for generating field reports
   */
  void attach_data(DataOut<dim> & data_out) const override;
  /* */
  void extract_solution_data
  (const typename dealii::DoFHandler<dim>::active_cell_iterator & cell,
   SolutionValues<dim,n_phases>                                 & solution_values) override;

  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
  get_pressure_saturation_function();

  // accessing private members
  const TrilinosWrappers::SparseMatrix & get_system_matrix();
  const TrilinosWrappers::MPI::Vector  & get_rhs_vector();
  const DoFHandler<dim>                & get_dof_handler();
  const FE_DGQ<dim>                    & get_fe() override;

 private:
  MPI_Comm                                  & mpi_communicator;
  parallel::distributed::Triangulation<dim> & triangulation;
  DoFHandler<dim>                             dof_handler;
  FE_DGQ<dim>                                 fe;
  const Model::Model<dim>                   & model;
  ConditionalOStream                        & pcout;

  // Matrices and vectors
  TrilinosWrappers::SparseMatrix              system_matrix;
  TrilinosWrappers::MPI::Vector               rhs_vector;

 public:
  TrilinosWrappers::MPI::Vector              solution,
                                             pressure,
                                             pressure_old;
  // std::vector<TrilinosWrappers::MPI::Vector> saturation_solution;
  std::vector<TrilinosWrappers::MPI::Vector> saturation,
                                             saturation_old;
  // partitioning
  IndexSet                      locally_owned_dofs, locally_relevant_dofs;

 private:
  const DoFHandler<dim>                     * p_solid_dof_handler;
  const TrilinosWrappers::MPI::Vector       * p_displacement;
  const TrilinosWrappers::MPI::Vector       * p_old_displacement;
  const FEValuesExtractors::Vector          * p_displacement_extractor;
  bool coupled_with_solid;

  Equations::IMPESPressure<n_phases>   cell_values;
  Equations::IMPESSaturation<n_phases> cell_values_saturation;
};


template <int dim, int n_phases>
SolverIMPES<dim,n_phases>::
SolverIMPES(MPI_Comm                                  & mpi_communicator_,
            parallel::distributed::Triangulation<dim> & triangulation_,
            const Model::Model<dim>                   & model_,
            ConditionalOStream                        & pcout_,
            Equations::IMPESPressure<n_phases>        & cell_values,
            Equations::IMPESSaturation<n_phases>      & cell_values_saturation)
    :
    mpi_communicator(mpi_communicator_),
    triangulation(triangulation_),
    dof_handler(triangulation_),
    fe(0), // since we want finite volumes
    model(model_),
    pcout(pcout_),
    // saturation_solution(model.n_phases()),
    saturation(model.n_phases()),
    saturation_old(model.n_phases() - 1),  // old solution n-1 phases
    coupled_with_solid(false),
    cell_values(cell_values),
    cell_values_saturation(cell_values_saturation)
{}  // eom


template <int dim, int n_phases>
SolverIMPES<dim,n_phases>::~SolverIMPES()
{
  dof_handler.clear();
}  // eom


template <int dim, int n_phases>
void SolverIMPES<dim,n_phases>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);
  locally_owned_dofs.clear();
  locally_relevant_dofs.clear();
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

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
    pressure_old.reinit(locally_relevant_dofs, mpi_communicator);
    pressure.reinit(locally_relevant_dofs, mpi_communicator);
    rhs_vector.reinit(locally_owned_dofs, locally_relevant_dofs,
                      mpi_communicator, /* omit-zeros= */ true);
    for (unsigned int p=0; p<model.n_phases(); ++p)
    {
      // saturation_solution[p].reinit(locally_owned_dofs, mpi_communicator);
      saturation[p].reinit(locally_relevant_dofs, mpi_communicator);
      // old solution stores only n-1 phases
      if (p < model.n_phases() - 1)
        saturation_old[p].reinit(locally_relevant_dofs, mpi_communicator);
    }
  }  // end setup vectors
} // eom



template <int dim, int n_phases>
void
SolverIMPES<dim,n_phases>::
assemble_pressure_system(const double time_step)
{
  assemble_flow_system
      <dim, TrilinosWrappers::MPI::Vector, TrilinosWrappers::SparseMatrix>
      (dof_handler, *p_solid_dof_handler,
       pressure, pressure_old, saturation,
       *p_displacement, *p_old_displacement, *p_displacement_extractor,
       cell_values,
       system_matrix, rhs_vector,
       time_step, model.n_phases(),
       coupled_with_solid, /* assemble_matrix = */ true);

  // // Only one integration point in FVM
  // QGauss<dim>   quadrature_formula(1);
  // QGauss<dim-1> face_quadrature_formula(1);

  // FEValues<dim> fe_values(fe, quadrature_formula, update_values);
  // FEValues<dim> fe_values_neighbor(fe, quadrature_formula, update_values);
  // // the following two objects only get geometry data
  // FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
  //                                  update_normal_vectors);
  // // We need JxW flag for subfaces since there is no
  // // method to determine sub face area in triangulation class
  // FESubfaceValues<dim> fe_subface_values(fe, face_quadrature_formula,
  //                                        update_normal_vectors |
  //                                        update_JxW_values);
  // FEValues<dim> * p_fe_values_solid = NULL;
  // if (coupled_with_solid)
  //   p_fe_values_solid = new FEValues<dim>(p_solid_dof_handler->get_fe(),
  //                                         quadrature_formula, update_gradients);
  // auto & fe_values_solid = * p_fe_values_solid;

  // const unsigned int dofs_per_cell = fe.dofs_per_cell;
  // const unsigned int n_q_points = quadrature_formula.size();
  // std::vector<types::global_dof_index>
  //     dof_indices(dofs_per_cell),
  //     dof_indices_neighbor(dofs_per_cell);

  // // objects to store local data
  // Tensor<1, dim>       normal;
  // std::vector<double>  p_values(n_q_points),
  //                      p_old_values(n_q_points);
  // std::vector<double>  div_u_values(n_q_points);
  // std::vector<double>  div_old_u_values(n_q_points);
  // std::vector< std::vector<double> >  s_values(model.n_phases()-1);
  // for (auto & c: s_values)
  //   c.resize(face_quadrature_formula.size());
  // // this one stores both saturation values and geomechanics
  // FluidEquations::ExtraValues extra_values;

  // const unsigned int q_point = 0;

  // typename DoFHandler<dim>::active_cell_iterator
  //     cell = dof_handler.begin_active(),
  //     // trick to place solid_cell in cell loop condition
  //     solid_cell = dof_handler.begin_active(),
  //     endc = dof_handler.end();

  // if (coupled_with_solid)
  //   solid_cell = p_solid_dof_handler->begin_active();

  // system_matrix = 0;
  // rhs_vector = 0;

  // for (; cell!=endc; ++cell, ++solid_cell)
  // {
  //   if (cell->is_locally_owned())
  //   {
  //     fe_values.reinit(cell);
  //     fe_values.get_function_values(pressure_old, p_old_values);
  //     fe_values.get_function_values(pressure_relevant, p_values);
  //     for (unsigned int c=0; c<model.n_phases() - 1; ++c)
  //     {
  //       fe_values.get_function_values(saturation_relevant[c], s_values[c]);
  //       extra_values.saturation[c] = s_values[c][q_point];
  //     }
  //     if (coupled_with_solid)
  //     {
  //       fe_values_solid.reinit(solid_cell);
  //       fe_values_solid[*p_displacement_extractor].
  //           get_function_divergences(*p_displacement, div_u_values);
  //       fe_values_solid[*p_displacement_extractor].
  //           get_function_divergences(*p_old_displacement, div_old_u_values);
  //       extra_values.div_u = div_u_values[q_point];
  //       extra_values.div_old_u = div_old_u_values[q_point];
  //     }

  //     // std::cout << "cell: " << i << std::endl;
  //     const double pressure_value = p_values[q_point];
  //     const double pressure_value_old = p_old_values[q_point];

  //     cell_values.update(cell, pressure_value, extra_values);
  //     cell_values.update_wells(cell);

  //     double matrix_ii = cell_values.get_matrix_cell_entry(time_step);
  //     // double rhs_i = cell_values.get_rhs_cell_entry(time_step,
  //     //                                               pressure_value_old,
  //     //                                               pressure_value);
  //     double rhs_i = cell_values.get_rhs_cell_entry(time_step,
  //                                                   pressure_value,
  //                                                   pressure_value_old);

  //     cell->get_dof_indices(dof_indices);
  //     const unsigned int i = dof_indices[q_point];
  //     for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
  //     {
  //       if (cell->at_boundary(f) == false)
  //       {
  //         if((cell->neighbor(f)->level() == cell->level() &&
  //             cell->neighbor(f)->has_children() == false) ||
  //            cell->neighbor_is_coarser(f))
  //         {
  //           const auto & neighbor = cell->neighbor(f);
  //           fe_values.reinit(neighbor);
  //           fe_face_values.reinit(cell, f);

  //           fe_values.get_function_values(pressure_relevant, p_values);
  //           for (unsigned int c=0; c<model.n_phases() - 1; ++c)
  //           {
  //             fe_values.get_function_values(saturation_relevant[c], s_values[c]);
  //             extra_values.saturation[c] = s_values[c][q_point];
  //           }
  //           if (coupled_with_solid)
  //           {
  //             fe_values_solid.reinit(solid_cell->neighbor(f));
  //             fe_values_solid[*p_displacement_extractor].
  //                 get_function_divergences(*p_displacement, div_u_values);
  //             fe_values_solid[*p_displacement_extractor].
  //                 get_function_divergences(*p_old_displacement, div_old_u_values);
  //             extra_values.div_u = div_u_values[q_point];
  //             extra_values.div_old_u = div_old_u_values[q_point];
  //           }
  //           const double p_neighbor = p_values[q_point];

  //           normal = fe_face_values.normal_vector(q_point);
  //           const double dS = cell->face(f)->measure();  // face area

  //           // assemble local matrix and distribute
  //           cell_values_neighbor.update(neighbor, p_neighbor, extra_values);
  //           cell_values.update_face_values(cell_values_neighbor, normal, dS);

  //           // distribute
  //           neighbor->get_dof_indices(dof_indices_neighbor);
  //           const unsigned int j = dof_indices_neighbor[q_point];
  //           const double face_entry = cell_values.get_matrix_face_entry();
  //           matrix_ii += face_entry;
  //           rhs_i += cell_values.get_rhs_face_entry(time_step);
  //           system_matrix.add(i, j, -face_entry);
  //         }
  //         else if ((cell->neighbor(f)->level() == cell->level()) &&
  //                  (cell->neighbor(f)->has_children() == true))
  //         {
  //           for (unsigned int subface=0;
  //                subface<cell->face(f)->n_children(); ++subface)
  //           {
  //             // compute parameters
  //             const auto & neighbor
  //                 = cell->neighbor_child_on_subface(f, subface);

  //             fe_values.reinit(neighbor);
  //             fe_subface_values.reinit(cell, f, subface);

  //             fe_values.get_function_values(pressure_relevant, p_values);
  //             for (unsigned int c=0; c<model.n_phases() - 1; ++c)
  //             {
  //               fe_values.get_function_values(saturation_relevant[c], s_values[c]);
  //               extra_values.saturation[c] = s_values[c][q_point];
  //             }
  //             if (coupled_with_solid)
  //             {
  //               const auto & solid_neighbor =
  //                   solid_cell->neighbor_child_on_subface(f, subface);
  //               fe_values_solid.reinit(solid_neighbor);
  //               fe_values_solid[*p_displacement_extractor].
  //                   get_function_divergences(*p_displacement, div_u_values);
  //               fe_values_solid[*p_displacement_extractor].
  //                   get_function_divergences(*p_old_displacement, div_old_u_values);
  //               extra_values.div_u = div_u_values[q_point];
  //               extra_values.div_old_u = div_old_u_values[q_point];
  //             }

  //             const double p_neighbor = p_values[q_point];
  //             normal = fe_subface_values.normal_vector(q_point);
  //             const double dS = fe_subface_values.JxW(q_point);

  //             // update neighbor
  //             cell_values_neighbor.update(neighbor, p_neighbor, extra_values);
  //             // update face values
  //             cell_values.update_face_values(cell_values_neighbor, normal, dS);

  //             // distribute
  //             neighbor->get_dof_indices(dof_indices_neighbor);
  //             const unsigned int j = dof_indices_neighbor[q_point];
  //             const double face_entry = cell_values.get_matrix_face_entry();
  //             matrix_ii += face_entry;
  //             rhs_i += cell_values.get_rhs_face_entry(time_step);
  //             system_matrix.add(i, j, -face_entry);
  //           }
  //         } // end case neighbor is finer

  //       } // end if face not at boundary
  //     }  // end face loop

  //     system_matrix.add(i, i, matrix_ii);
  //     rhs_vector[i] += rhs_i;
  //   } // end local cells
  // } // end cells loop
  // system_matrix.compress(VectorOperation::add);
  // rhs_vector.compress(VectorOperation::add);
  // delete p_fe_values_solid;
}  // eom



template <int dim, int n_phases>
void
SolverIMPES<dim,n_phases>::
solve_saturation_system(const double time_step)
{
  assemble_flow_system
      <dim, TrilinosWrappers::MPI::Vector, TrilinosWrappers::SparseMatrix>
      (dof_handler, *p_solid_dof_handler,
       pressure, pressure_old, saturation,
       *p_displacement, *p_old_displacement, *p_displacement_extractor,
       cell_values_saturation,
       /*not used*/ system_matrix, /* rhs_vector = */ rhs_vector,
       time_step, model.n_phases(),
       coupled_with_solid, /* assemble_matrix = */ false);

  const int phase = 0;
  std::pair<double,double> saturation_limits =
      model.get_saturation_limits(phase);

  for (const auto & dof : locally_owned_dofs)
  {
    const double Sw_old = saturation[0][dof];
    double increment = rhs_vector[dof];

    // assert that we are in bounds
    if (Sw_old + increment > saturation_limits.second)
          increment = saturation_limits.second - Sw_old;
    else if (Sw_old + increment < saturation_limits.first)
          increment = saturation_limits.first - Sw_old;

    solution[dof] = Sw_old + increment;
  } // end dof loop

  solution.compress(VectorOperation::insert);
  saturation[0] = solution;
} // eom


template <int dim, int n_phases>
unsigned int
SolverIMPES<dim,n_phases>::solve_pressure_system()
{
  double tol = 1e-10*rhs_vector.l2_norm();
  if (tol == 0.0)
    tol = 1e-10;
  SolverControl solver_control(1000, tol);

  if (model.linear_solver_fluid == Model::LinearSolverType::CG)
  { // iterative solver
    TrilinosWrappers::SolverCG::AdditionalData additional_data_cg;
    TrilinosWrappers::SolverCG
        solver(solver_control, additional_data_cg);
    TrilinosWrappers::PreconditionAMG::AdditionalData additional_data_amg;
    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix, additional_data_amg);
    solver.solve(system_matrix, solution, rhs_vector, preconditioner);
  }
  else if (model.linear_solver_solid == Model::LinearSolverType::Direct)
  { // direct solver
    TrilinosWrappers::SolverDirect
        solver(solver_control, TrilinosWrappers::SolverDirect::AdditionalData());
    solver.solve(system_matrix, solution, rhs_vector);
  }

  pressure = solution;

  return solver_control.last_step();
} // eom



template <int dim, int n_phases>
unsigned int
SolverIMPES<dim,n_phases>::solve_time_step(const double time_step)
{
  assemble_pressure_system(time_step);
  unsigned int n_steps = solve_pressure_system();

  if (n_phases > 1)
    solve_saturation_system(time_step);

  return n_steps;
}



template <int dim, int n_phases>
void
SolverIMPES<dim,n_phases>::
set_coupling(const DoFHandler<dim>               & solid_dof_handler,
             const TrilinosWrappers::MPI::Vector & displacement_vector,
             const TrilinosWrappers::MPI::Vector & old_displacement_vector,
             const FEValuesExtractors::Vector    & extractor)
{
  p_solid_dof_handler      = & solid_dof_handler;
  p_displacement           = & displacement_vector;
  p_old_displacement       = & old_displacement_vector;
  p_displacement_extractor = & extractor;
  coupled_with_solid = true;
}  // eom



template <int dim, int n_phases>
const TrilinosWrappers::SparseMatrix&
SolverIMPES<dim,n_phases>::get_system_matrix()
{
  return system_matrix;
}  // eom



template <int dim, int n_phases>
const TrilinosWrappers::MPI::Vector&
SolverIMPES<dim,n_phases>::get_rhs_vector()
{
  return rhs_vector;
}  // eom



template <int dim, int n_phases>
const DoFHandler<dim> &
SolverIMPES<dim,n_phases>::get_dof_handler()
{
  return dof_handler;
}  // eom



template <int dim, int n_phases>
const FE_DGQ<dim> &
SolverIMPES<dim,n_phases>::get_fe()
{
  return fe;
}  // eom



template <int dim, int n_phases>
void
SolverIMPES<dim,n_phases>::attach_data(DataOut<dim> & data_out) const
{
  // data_out.attach_dof_handler(dof_handler);
  // // scale pressure by bar/psi/whatever
  // Output::ScaleOutputVector<dim> pressure_scaler(Keywords::pressure_vector,
  //                                                model.units.pressure());
  // // data_out.add_data_vector(pressure_relevant, Keywords::pressure_vector,
  // //                          DataOut<dim>::type_dof_data);
  // data_out.add_data_vector(pressure, pressure_scaler);
  // data_out.add_data_vector(saturation, Keywords::saturation_water_vector,
  //                          DataOut<dim>::type_dof_data);
}  // end attach_data



template<int dim, int n_phases>
FEFunction::FEFunction<dim, TrilinosWrappers::MPI::Vector>
SolverIMPES<dim,n_phases>::get_pressure_saturation_function()
{
  return FEFunction::FEFunctionPS<dim,TrilinosWrappers::MPI::Vector>(dof_handler,
                                                                     pressure,
                                                                     saturation);
}  // end get_pressure_saturation_function



template<int dim, int n_phases>
void
SolverIMPES<dim,n_phases>::extract_solution_data
(const typename DoFHandler<dim>::active_cell_iterator & cell,
 SolutionValues<dim,n_phases>                         & solution_values)
{
  throw(ExcNotImplemented());
} // eom

}  // end of namespace

} // end wings

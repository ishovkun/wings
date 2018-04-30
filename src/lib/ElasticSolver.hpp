#pragma once

#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>  // interpolate_boundary_values

// custom modules
#include <SolidSolverBase.hpp>
#include <Model.hpp>
#include <Math.hpp>


namespace Wings {

namespace SolidSolvers
{
using namespace dealii;


template <int dim, int n_phases>
class ElasticSolver : public SolidSolverBase<dim,n_phases>
{
 public:
  // Constructor
  ElasticSolver(MPI_Comm                                  & mpi_communicator,
                parallel::distributed::Triangulation<dim> & triangulation,
                const Model::Model<dim>                   & model,
                ConditionalOStream                        & pcout);
  // Destructor
  ~ElasticSolver();
  /* setup degrees of freedom for the current triangulation
   * and allocate memory for solution vectors */
  void setup_dofs();
  // Fill system matrix and rhs vector
  void assemble_system(const TrilinosWrappers::MPI::Vector & pressure_vector);
  // solve linear system syste_matrix*solution= rhs_vector
  // returns the number of solver steps
  unsigned int solve_linear_system();
  // give solver access to fluid dofs
  void set_coupling(const DoFHandler<dim> & fluid_dof_handler);
  // assemble_system() and solve()
  unsigned int solve_time_step(const double /*time_step*/) override;
  // for field output
  void attach_data(DataOut<dim> & data_out) const override;
  // for probe
  void extract_solution_data
  (const typename DoFHandler<dim>::active_cell_iterator & cell,
   SolutionValues<dim,n_phases>                         & solution_values) override;

  // accessing private members
  const TrilinosWrappers::SparseMatrix & get_system_matrix();
  const TrilinosWrappers::MPI::Vector  & get_rhs_vector();
  const DoFHandler<dim>                & get_dof_handler() override;


 private:
  MPI_Comm                                  & mpi_communicator;
  parallel::distributed::Triangulation<dim> & triangulation;
  DoFHandler<dim>                             dof_handler;
  FESystem<dim>                               fe;
  const Model::Model<dim>                   & model;
  ConditionalOStream                        & pcout;
  // Matrices and vectors
  TrilinosWrappers::SparseMatrix         system_matrix;
  TrilinosWrappers::MPI::Vector          rhs_vector;
  const DoFHandler<dim>                * p_fluid_dof_handler;
  ConstraintMatrix                       constraints;

 public:
  // solution vectors
  TrilinosWrappers::MPI::Vector solution, old_solution;
  TrilinosWrappers::MPI::Vector relevant_solution;
  // partitioning
  IndexSet                      locally_owned_dofs, locally_relevant_dofs;
};



template <int dim, int n_phases>
ElasticSolver<dim,n_phases>::
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



template <int dim, int n_phases>
ElasticSolver<dim,n_phases>::~ElasticSolver()
{
  dof_handler.clear();
} // eom



template <int dim, int n_phases>
void
ElasticSolver<dim,n_phases>::set_coupling(const DoFHandler<dim> & fluid_dof_handler)
{
  p_fluid_dof_handler = & fluid_dof_handler;
} // eom



template <int dim, int n_phases>
void
ElasticSolver<dim,n_phases>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);

  { // partitioning
    locally_owned_dofs.clear();
    locally_relevant_dofs.clear();
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);
  }
  { // constraints
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // add dirichlet BC's to constraints
    std::vector<ComponentMask> mask(dim);
    for (unsigned int comp=0; comp<dim; ++comp)
    {
      FEValuesExtractors::Scalar extractor(comp);
      mask[comp] = fe.component_mask(extractor);
    }
    int n_dirichlet_conditions = model.solid_dirichlet_labels.size();

    for (int cond=0; cond<n_dirichlet_conditions; ++cond)
    {
      int component = model.solid_dirichlet_components[cond];
      double dirichlet_value = model.solid_dirichlet_values[cond];
      VectorTools::interpolate_boundary_values
          (dof_handler,
           model.solid_dirichlet_labels[cond],
           ConstantFunction<dim>(dirichlet_value, dim),
           constraints,
           mask[component]);
    }

    constraints.close();
  }
  { // system matrix
    system_matrix.clear();
    TrilinosWrappers::SparsityPattern
        sparsity_pattern(locally_owned_dofs, mpi_communicator);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern,
                                    constraints,
                                    /* keep_constrained_dofs =  */ false);
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



template <int dim, int n_phases>
void
ElasticSolver<dim,n_phases>::
assemble_system(const TrilinosWrappers::MPI::Vector & pressure_vector)
{
  const auto &  fluid_fe = p_fluid_dof_handler->get_fe();

  QGauss<dim>   fvm_quadrature_formula(1);
  QGauss<dim>   quadrature_formula(fe.degree + 1);
  QGauss<dim-1> face_quadrature_formula(fe.degree + 1);

  FEValues<dim>     fe_values(fe, quadrature_formula,
                              update_values | update_gradients |
                              update_JxW_values);
  FEValues<dim>     fluid_fe_values(fluid_fe,
                                    fvm_quadrature_formula,
                                    update_values);
  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   update_values |
                                   update_normal_vectors |
                                   update_JxW_values);

  // we need this because FeSystem class is weird
  // we use this extractor to extract all (displacement) dofs
  const FEValuesExtractors::Vector displacement(0);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();
  const unsigned int n_neumann_conditions = model.solid_neumann_labels.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  FullMatrix<double>                   cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>                       cell_rhs(dofs_per_cell);
  std::vector< Tensor<2,dim> > eps_u(dofs_per_cell);
  std::vector< Tensor<2,dim> > sigma_u(dofs_per_cell);
  std::vector< Tensor<2,dim> > grad_xi_u(dofs_per_cell);
  std::vector<double> 				 p_values(1); // 1 point since FVM
  Tensor<2,dim> identity_tensor = Math::get_identity_tensor<dim>();


  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end(),
      fluid_cell = p_fluid_dof_handler->begin_active();

  system_matrix = 0;
  rhs_vector = 0;

  for (; cell!=endc; ++cell, ++fluid_cell)
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fluid_fe_values.reinit(fluid_cell);

      cell_matrix = 0;
      cell_rhs = 0;

      fluid_fe_values.get_function_values(pressure_vector, p_values);
      const double p_value = p_values[0];

			const double E = model.get_young_modulus->value(cell->center(), 0);
			const double nu = model.get_poisson_ratio->value(cell->center(), 0);
			const double lame_constant = E*nu/((1.+nu)*(1.-2*nu));
			const double shear_modulus = 0.5*E/(1.+nu);
      const double alpha = model.get_biot_coefficient();
      // pcout << "E = " << E  << std::endl;
      // pcout << "nu = " << nu  << std::endl;

      for (unsigned int q=0; q<n_q_points; ++q)
      {
        // compute stresses and strains for each local dof
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
					grad_xi_u[k] = fe_values[displacement].gradient(k, q);
					eps_u[k] 		 = 0.5*(grad_xi_u[k] + transpose(grad_xi_u[k]));
          sigma_u[k] =
              lame_constant*trace(eps_u[k])*identity_tensor +
              2*shear_modulus*eps_u[k];
        } // end k loop

        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            cell_matrix(i, j) +=
                scalar_product(sigma_u[j], eps_u[i]) *
                fe_values.JxW(q);
          } // end j loop

          cell_rhs(i)  +=
              alpha*p_value*trace(grad_xi_u[i])*fe_values.JxW(q);
        }  // end i loop
      }  // end q loop

      // impose neumann BC's
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
        {
          unsigned int face_boundary_id = cell->face(f)->boundary_id();
          // pcout << "at boundary" << cell->center()
          //       << "\t" << "face = " << f
          //       << "\t" << "id = " << face_boundary_id
          //       << std::endl;
          fe_face_values.reinit(cell, f);

          // loop through input neumann labels
          for (unsigned int l=0; l<n_neumann_conditions; ++l)
          {
            const unsigned int id = model.solid_neumann_labels[l];

            if (face_boundary_id == id)
            {
              // pcout << "at neumann boundary " << id << std::endl;

              for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                // pcout << "trying component " << component_i << std::endl;

                const unsigned int neumann_component =
                    model.solid_neumann_components[l];

                if (component_i == neumann_component)
                  for (unsigned int q=0; q<n_face_q_points; ++q)
                  {
                    const double neumann_value =
                        model.solid_neumann_values[l] *
                        fe_face_values.normal_vector(q)[component_i];

                    // pcout << "adding stuff " << neumann_value << std::endl;

                    cell_rhs(i) +=
                        fe_face_values.shape_value(i, q) *
                        neumann_value *
                        fe_face_values.JxW(q);
                  }  // end of q_point loop
              }  // end of i loop
            } // end at neumann boundary

          }  // end loop neumann labels
        } // end face loop
      // end Neumann BC's

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global
          (cell_matrix, cell_rhs, local_dof_indices,
           system_matrix, rhs_vector);
    } // end cell loop

  system_matrix.compress(VectorOperation::add);
  rhs_vector.compress(VectorOperation::add);
}  // eom



template<int dim, int n_phases>
unsigned int
ElasticSolver<dim,n_phases>::solve_linear_system()
{
  // setup CG solver
  TrilinosWrappers::SolverCG::AdditionalData data_cg;
  double tol = 1e-10*rhs_vector.l2_norm();
  if (tol == 0.0)
    tol = 1e-10;
  SolverControl solver_control(1000, tol);

  if (model.linear_solver_solid == Model::LinearSolverType::CG)
  {
    TrilinosWrappers::SolverCG solver(solver_control, data_cg);

    // setup AMG preconditioner
    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data_amg;
    // data_amg.constant_modes = constant_modes;
    data_amg.elliptic = true;
    data_amg.higher_order_elements = true;
    data_amg.smoother_sweeps = 2;
    data_amg.aggregation_threshold = 0.02;
    preconditioner.initialize(system_matrix, data_amg);

    // solve linear system
    solver.solve(system_matrix, solution, rhs_vector, preconditioner);
  }
  else if (model.linear_solver_solid == Model::LinearSolverType::Direct)
  { // direct solver
    TrilinosWrappers::SolverDirect
        solver(solver_control, TrilinosWrappers::SolverDirect::AdditionalData());
    solver.solve(system_matrix, solution, rhs_vector);
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }

  constraints.distribute(solution);

  return solver_control.last_step();
}  // end solve



template <int dim, int n_phases>
const TrilinosWrappers::SparseMatrix &
ElasticSolver<dim,n_phases>::get_system_matrix()
{
  return system_matrix;
}  // end get_syste_matrix



template <int dim, int n_phases>
const DoFHandler<dim> &
ElasticSolver<dim,n_phases>::get_dof_handler()
{
  return dof_handler;
}  // eom



template <int dim, int n_phases>
unsigned int ElasticSolver<dim,n_phases>::solve_time_step(const double)
{
  throw(ExcNotImplemented());
}  // eom



template <int dim, int n_phases>
void ElasticSolver<dim,n_phases>::attach_data(DataOut<dim> & data_out) const
{
  throw(ExcNotImplemented());
}  // eom



template <int dim, int n_phases>
void ElasticSolver<dim,n_phases>::
extract_solution_data
(const typename DoFHandler<dim>::active_cell_iterator & cell,
 SolutionValues<dim,n_phases>                         & solution_values)
{
  throw(ExcNotImplemented());
}  // eom

} // end of namespace

} // end wings

#pragma once

// #include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/fe_values.h>
// vectors and matrices
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/base/config.h>  // for numbers::is_nan

// to print sparsity pattern, remove later
#include <fstream>

#include <deal.II/base/quadrature_lib.h>
// #include <deal.II/base/function.h>
// #include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/grid_out.h>
// #include <deal.II/grid/grid_refinement.h>
// #include <deal.II/grid/tria_accessor.h>
// #include <deal.II/grid/tria_iterator.h>
// #include <deal.II/dofs/dof_accessor.h>
// #include <deal.II/numerics/data_out.h>
// #include <deal.II/fe/mapping_q1.h>

// #include <deal.II/meshworker/dof_info.h>
// #include <deal.II/meshworker/integration_info.h>
// #include <deal.II/meshworker/simple.h>
// #include <deal.II/meshworker/loop.h>

#include <deal.II/base/utilities.h>
// #include <deal.II/base/function.h>
// #include <deal.II/base/tensor.h>

// Trilinos stuff
// #include <deal.II/lac/generic_linear_algebra.h>
// #include <deal.II/lac/solver_gmres.h>
// #include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
// #include <deal.II/lac/trilinos_block_sparse_matrix.h>
// #include <deal.II/lac/trilinos_block_vector.h>

// DOF stuff
#include <deal.II/distributed/tria.h>
// #include <deal.II/dofs/dof_handler.h>
// #include <deal.II/dofs/dof_renumbering.h>
// #include <deal.II/dofs/dof_accessor.h>
// #include <deal.II/dofs/dof_tools.h>

// dealii fem modules
// #include <deal.II/fe/fe_values.h>
// #include <deal.II/numerics/vector_tools.h>
// #include <deal.II/fe/fe_system.h>
// #include <deal.II/lac/sparsity_tools.h>

// Custom modules
#include <DataBase.hpp>
#include <Wellbore.hpp>
#include <CellValues.hpp>

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
                   const Data::DataBase<dim>                 &data_,
                   ConditionalOStream                        &pcout_);
    ~PressureSolver();

    void setup_dofs();
    void assemble_system(CellValues::CellValuesBase<dim> &cell_data,
                         CellValues::CellValuesBase<dim> &neighbor_data,
                         const double time_step);
    unsigned int solve();
    // accessing private members
    const SparseMatrix<double>& get_system_matrix();
    const Vector<double>&       get_rhs_vector();
    const DoFHandler<dim> &     get_dof_handler();
    const FE_DGQ<dim> &         get_fe();

  private:
    MPI_Comm                                  &mpi_communicator;
    parallel::distributed::Triangulation<dim> &triangulation;
    DoFHandler<dim>                           dof_handler;
    FE_DGQ<dim>                               fe;
    const Data::DataBase<dim>                 &data;
    ConditionalOStream                        &pcout;

    // Matrices and vectors
    SparsityPattern                           sparsity_pattern;
    SparseMatrix<double>                      system_matrix;

    // SparsityPattern
    // typedef MeshWorker::DoFInfo<dim> DoFInfo;
    // typedef MeshWorker::IntegrationInfo<dim> CellInfo;

  public:
    Vector<double>                solution, solution_old, rhs_vector;
    // std::vector< Vector<double> > permeability;
    // Vector<double>       perm;
  };


  template <int dim>
  PressureSolver<dim>::
  PressureSolver(MPI_Comm                                  &mpi_communicator_,
                 parallel::distributed::Triangulation<dim> &triangulation_,
                 const Data::DataBase<dim>                 &data_,
                 ConditionalOStream                        &pcout_)
    :
    mpi_communicator(mpi_communicator_),
    triangulation(triangulation_),
    dof_handler(triangulation_),
    fe(0), // since we want finite volumes
    data(data_),
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
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    // std::ofstream out ("sparsity_pattern1.svg");
    // sparsity_pattern.print_svg (out);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    solution_old.reinit(dof_handler.n_dofs());
    rhs_vector.reinit(dof_handler.n_dofs());
  } // eom


  template <int dim>
  void
  PressureSolver<dim>::assemble_system(CellValues::CellValuesBase<dim> &cell_values,
                                       CellValues::CellValuesBase<dim> &neighbor_values,
                                       const double time_step)
  {
    // Only one integration point in FVM
    QGauss<dim>       quadrature_formula(1);
    QGauss<dim-1>     face_quadrature_formula(1);

    FEValues<dim> fe_values(fe, quadrature_formula, update_values);
    FEValues<dim> fe_values_neighbor(fe, quadrature_formula,
                                     update_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_normal_vectors);
    // We need subface values for subfaces since there is no
    // method to determine sub face area in triangulation class
    FESubfaceValues<dim> fe_subface_values(fe, face_quadrature_formula,
                                           update_normal_vectors |
                                           update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index>
      dof_indices(dofs_per_cell),
      dof_indices_neighbor(dofs_per_cell);

    Tensor<1, dim>    dx_ij, normal;
    std::vector<double>    p_old_values(quadrature_formula.size());

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    system_matrix = 0;
    rhs_vector = 0;

    for (; cell!=endc; ++cell)
    {
      cell->get_dof_indices(dof_indices);
      unsigned int i = dof_indices[0];
       // std::cout << "i = " << i << "\t" << cell->center() << std::endl;
      fe_values.reinit(cell);
      fe_values.get_function_values(solution, p_old_values);
      // std::cout << "cell: " << i << std::endl;
      // double p_i = solution[i];
      // double p_old = solution_old[i];
      double p_old = p_old_values[0];

      cell_values.update(cell);

      const double B_ii = cell_values.get_mass_matrix_entry();

      const double J_i = cell_values.J;
      const double Q_i = cell_values.Q;
      // // Wells
      // double Q_i = 0;
      // double J_i = 0;
      // for (auto & well : data.wells)
      // {
      //   Q_i += well.get_rate_water(cell);
      //   J_i += well.get_productivity(cell);
      // } // end well loop


      double matrix_ii = B_ii/time_step + J_i;
      double rhs_i = B_ii/time_step*p_old + Q_i;

      unsigned int j = 0;
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (cell->at_boundary(f) == false)
        {
          if((cell->neighbor(f)->level() == cell->level() &&
             cell->neighbor(f)->has_children() == false) ||
             cell->neighbor_is_coarser(f))
          {
            cell->neighbor(f)->get_dof_indices(dof_indices_neighbor);
            fe_face_values.reinit(cell, f);
            normal = fe_face_values.normal_vector(0); // 0 is gauss point
            j = dof_indices_neighbor[0];
            const double dS = cell->face(f)->measure();  // face area
            dx_ij = cell->neighbor(f)->center() - cell->center();
            neighbor_values.update(cell->neighbor(f));
            // assemble local matrix and distribute
            cell_values.update_face_values(neighbor_values, dx_ij, normal, dS);
            matrix_ii += cell_values.T_face;
            rhs_i += cell_values.G_face;
            system_matrix.add(i, j, -cell_values.T_face);
          }
          else if ((cell->neighbor(f)->level() == cell->level()) &&
                   (cell->neighbor(f)->has_children() == true))
          {
            for (unsigned int subface=0;
                 subface<cell->face(f)->n_children(); ++subface)
            {
              // compute parameters
              const auto & neighbor_child
                  = cell->neighbor_child_on_subface(f, subface);
              neighbor_child->get_dof_indices(dof_indices_neighbor);
              j = dof_indices_neighbor[0];
              fe_subface_values.reinit(cell, f, subface);
              normal = fe_subface_values.normal_vector(0); // 0 is gauss point
              neighbor_values.update(neighbor_child);
              const double dS = fe_subface_values.JxW(0);
              dx_ij = neighbor_child->center() - cell->center();
              // assemble local matrix and distribute
              cell_values.update_face_values(neighbor_values, dx_ij, normal, dS);
              matrix_ii += cell_values.T_face;
              rhs_i += cell_values.G_face;
              system_matrix.add(i, j, -cell_values.T_face);
            }
          }
          // else if (cell->neighbor_is_coarser(f))
          // {
          //   // compute dh on cell
          //   // dx is not between cell centers i think
          //   // get_cell_data(neighbor);
          //   // compute_intercell_data();
          //   dx_ij = cell->neighbor(f)->center() - cell->center();
          // }

        } // end if face not at boundary
      }  // end face loop

      system_matrix.add(i, i, matrix_ii);
      rhs_vector[i] += rhs_i;

      // std::cout << "------------------------------\n";
    } // end cell loop

  } // eom


  template <int dim>
  unsigned int
  PressureSolver<dim>::solve()
  {
    double tol = 1e-10*rhs_vector.l2_norm();
    if (tol == 0.0)
      tol = 1e-10;
    SolverControl solver_control(1000, tol);
    SolverCG<> solver(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
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
  const SparseMatrix<double>&
  PressureSolver<dim>::get_system_matrix()
  {
    return system_matrix;
  }  // eom


  template <int dim>
  const Vector<double>&
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

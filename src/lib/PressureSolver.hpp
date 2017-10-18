#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/fe_values.h>
// vectors and matrices
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
// for numbers::is_nan
#include <deal.II/base/config.h>

// to print sparsity pattern, remove later
#include <fstream>

// #include <deal.II/base/quadrature_lib.h>
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

// #include <deal.II/base/utilities.h>
// #include <deal.II/base/quadrature_lib.h>
// #include <deal.II/base/timer.h>
// #include <deal.II/base/function.h>
// #include <deal.II/base/tensor.h>

// Trilinos stuff
// #include <deal.II/lac/generic_linear_algebra.h>
// #include <deal.II/lac/solver_gmres.h>
// #include <deal.II/lac/trilinos_solver.h>
// #include <deal.II/lac/trilinos_block_sparse_matrix.h>
// #include <deal.II/lac/trilinos_block_vector.h>

// DOF stuff
// #include <deal.II/distributed/tria.h>
// #include <deal.II/dofs/dof_handler.h>
// #include <deal.II/dofs/dof_renumbering.h>
// #include <deal.II/dofs/dof_accessor.h>
// #include <deal.II/dofs/dof_tools.h>

// dealii fem modules
// #include <deal.II/fe/fe_values.h>
// #include <deal.II/numerics/vector_tools.h>
// #include <deal.II/fe/fe_system.h>
// #include <deal.II/lac/sparsity_tools.h>


namespace FluidSolvers
{
	using namespace dealii;


  template <int dim>
  class PressureSolver
  {
  public:
    PressureSolver(Triangulation<dim> &triangulation);
    ~PressureSolver();

    void setup_system();
    void assemble_system();
    void solve();
    void print_system_matrix();

    Vector<double>       solution;

  private:
    DoFHandler<dim>      dof_handler;
    FE_DGQ<dim>          fe;

    // Matrices and vectors
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    // Vector<double>       right_hand_side;

    // SparsityPattern
    // typedef MeshWorker::DoFInfo<dim> DoFInfo;
    // typedef MeshWorker::IntegrationInfo<dim> CellInfo;
  };


  template <int dim>
  PressureSolver<dim>::PressureSolver(Triangulation<dim> &triangulation)
    :
    dof_handler(triangulation),
    fe(0)  // since we want finite volumes
  {}  // eom


  template <int dim>
  PressureSolver<dim>::~PressureSolver()
  {
    dof_handler.clear();
  }  // eom


  template <int dim>
  void PressureSolver<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    std::ofstream out ("sparsity_pattern1.svg");
    sparsity_pattern.print_svg (out);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    // solution_old.reinit(dof_handler.n_dofs());
    // right_hand_side.reinit (dof_handler.n_dofs());
  } // eom


  template <int dim>
  void PressureSolver<dim>::assemble_system()
  {
    // Only one integration point in FVM
    QGauss<dim>       quadrature_formula(1);
    QGauss<dim-1>     face_quadrature_formula(1);

    // FEValues<dim> fe_values(fe, quadrature_formula, update_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                 update_normal_vectors);

    Tensor<1, dim>    dx_j, normal;
    Tensor<1, dim>    perm;
    perm[0] = 1;
    perm[1] = 1;
    double visc = 1;
    // Tensor <1,dim> normal_vector;

    typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

    system_matrix = 0;

	  for (; cell!=endc; ++cell)
    {
      unsigned int cell_index = cell->active_cell_index();
      double p_i = solution[cell_index];
      // std::cout << "cell: " << cell_index << "\t";
      // std::cout << "cell pressure: " << solution[cell_index] << std::endl;
      double matrix_ii = 0;
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        unsigned int neighbor_index = cell->neighbor_index(f);
        unsigned int no_neighbor_index = -1;

        if(neighbor_index != no_neighbor_index) // if this neighbor exists
        {
          // CHECK IF NEIGHBOR IS NOT REFINED, OTHERWISE WRITE CODE!!!!!
          // if neighbor is not at the maximum refinement level
          fe_face_values.reinit(cell, f);
          double face_measure = cell->face(f)->measure();
          double p_j = solution[neighbor_index];
          normal = fe_face_values.normal_vector(0); // 0 is gauss point
          dx_j = cell->neighbor(f)->center() - cell->center();
          double mult = 0;
          // std::cout << "dx: " << dx_j[0] << "\t" << dx_j[1] << std::endl;
          for (int d=0; d<dim; d++)
          {
            if (!numbers::is_finite(dx_j[d]))
              mult = -1./visc*(perm[d]*normal[d]/dx_j[d])*face_measure;
          }
          matrix_ii += mult*p_i;
          double matrix_ij = mult*p_j;
          // std::cout << "neighbor: " << neighbor_index << std::endl;
          // std::cout << solution[neighbor_index] << std::endl;
          // std::cout << "measure " << face_measure << std::endl;
          // std::cout << "normals: "
          //           << fe_face_values.normal_vector(0)[0] << "\t"
          //           << fe_face_values.normal_vector(0)[1]
          //           << std::endl;
          system_matrix.add(cell_index, neighbor_index, matrix_ij);
        } // end face loop
        // system_matrix[cell_index, cell_index] += matrix_ii;
        system_matrix.add(cell_index, cell_index, matrix_ii);
      }  // end face loop

      // std::cout << "\n";
    } // end cell loop

  } // eom


  template <int dim>
  void PressureSolver<dim>::print_system_matrix()
  {
    for (unsigned int i=0; i<dof_handler.n_dofs(); i++) {
      for (unsigned int j=0; j<dof_handler.n_dofs(); j++) {
        std::cout << system_matrix(i, j) << "\t";
      }
      std::cout << std::endl;
    }
  } // eom
}  // end of namespace

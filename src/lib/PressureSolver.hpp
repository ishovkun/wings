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

// Custom modules
#include <DataBase.hpp>


namespace FluidSolvers
{
	using namespace dealii;


  template <int dim>
  class PressureSolver
  {
  public:
    PressureSolver(const Triangulation<dim>  &triangulation,
                   const Data::DataBase<dim> &data_);
    ~PressureSolver();

    void setup_system();
    void assemble_system(const double time_step);
    void solve();
    void print_system_matrix();


  private:
    double get_transmissibility(const Tensor<1,dim> &perm,
                                const double        visc,
                                const double        volume_factor,
                                const Tensor<1,dim> &normal_vector,
                                const Tensor<1,dim> &dx,
                                const double        dS) const;

    double get_cell_mass_matrix(const double cell_volume,
                                const double volume_factor,
                                const double porosity,
                                const double compressibility) const;

    void harmonic_mean(const Tensor<1,dim> &perm_1,
                       const Tensor<1,dim> &perm_2,
                       Tensor<1,dim>       &out) const;

    double arithmetic_mean(const double x1,
                           const double x2) const;

  private:
    DoFHandler<dim>            dof_handler;
    FE_DGQ<dim>                fe;
    const Data::DataBase<dim>  &data;

    // Matrices and vectors
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    // Vector<double>       right_hand_side;

    // SparsityPattern
    // typedef MeshWorker::DoFInfo<dim> DoFInfo;
    // typedef MeshWorker::IntegrationInfo<dim> CellInfo;

  public:
    Vector<double>                solution, solution_old, rhs_vector;
    // std::vector< Vector<double> > permeability;
    // Vector<double>       perm;
  };


  template <int dim>
  PressureSolver<dim>::PressureSolver(const Triangulation<dim>  &triangulation,
                                      const Data::DataBase<dim> &data_)
    :
    dof_handler(triangulation),
    fe(0),                          // since we want finite volumes
    data(data_)
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
    solution_old.reinit(dof_handler.n_dofs());
    rhs_vector.reinit(dof_handler.n_dofs());
  } // eom


  template <int dim>
  double PressureSolver<dim>::get_cell_mass_matrix(const double cell_volume,
                                                   const double volume_factor,
                                                   const double porosity,
                                                   const double compressibility) const
  {
    double B_ii = cell_volume/volume_factor*(porosity*compressibility);
    return B_ii;
  }  // eom


  template <int dim>
  double PressureSolver<dim>::get_transmissibility(const Tensor<1,dim> &perm,
                                                   const double        visc,
                                                   const double        volume_factor,
                                                   const Tensor<1,dim> &normal_vector,
                                                   const Tensor<1,dim> &dx,
                                                   const double        dS) const
  {
    double distance = dx.norm(); // to normalize
    if (distance == 0)
      return 0.0;

    double T = 0;
    for (int d=0; d<dim; ++d)
    {
      if (abs(dx[d]/distance) > 1e-10)
      {
        T += 1./visc/volume_factor*(perm[d]*normal_vector[d]/dx[d])*dS;
      }
    }

    return T;
  }  // eom

  template <int dim>
  void PressureSolver<dim>::harmonic_mean(const Tensor<1,dim> &perm_1,
                                          const Tensor<1,dim> &perm_2,
                                          Tensor<1,dim>       &out) const
  {
    for (int d=0; d<dim; ++d){
      if (perm_1[d] == 0 || perm_2[d] == 0)
        out[d] = 0;
      else
        out[d] = 2/(1./perm_1[d] + 1./perm_2[d]);
    }
  }  // eom


  template <int dim>
  double PressureSolver<dim>::arithmetic_mean(const double x1,
                                              const double x2) const
  {
    return 0.5*(x1+x2);
  }  // eom


  template <int dim>
  void PressureSolver<dim>::assemble_system(const double time_step)
  {
    // Only one integration point in FVM
    QGauss<dim>       quadrature_formula(1);
    QGauss<dim-1>     face_quadrature_formula(1);

    // FEValues<dim> fe_values(fe, quadrature_formula, update_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                 update_normal_vectors);

    Tensor<1, dim>    dx_ij, normal;
    Tensor<1, dim>    perm_i, perm_j, perm_ij;
    // Tensor <1,dim> normal_vector;

    typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
      neighbor_cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

    system_matrix = 0;
    rhs_vector = 0;

	  for (; cell!=endc; ++cell)
    {
      unsigned int i = cell->active_cell_index();
      // std::cout << "cell: " << i << std::endl;
      const double dV = cell->measure();
      // double p_i = solution[i];
      double p_old = solution_old[i];

      // Cell properties
      // data.get_permeability(i, perm_i);
      for (int d=0; d<dim; d++)
        perm_i[d] = data.get_permeability->value(cell->center(), d);
      const double mu_i = data.get_viscosity();
      const double volume_factor_i = data.get_volume_factor();
      const double poro_i = data.get_porosity->value(cell->center());
      const double fcomp_i = data.get_compressibility();

      // Cell mass matrix
      double B_ii = get_cell_mass_matrix(dV, volume_factor_i, poro_i, fcomp_i);

      // std::cout << "cell pressure: " << solution[cell_index] << std::endl;

      double matrix_ii = B_ii/time_step;
      double rhs_i = B_ii/time_step*p_old;

      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        unsigned int j = cell->neighbor_index(f);
        unsigned int no_neighbor_index = -1;

        if(j != no_neighbor_index) // if this neighbor exists
        {
          // CHECK IF NEIGHBOR IS NOT REFINED, OTHERWISE WRITE CODE!!!!!
          // if neighbor is not at the maximum refinement level
          fe_face_values.reinit(cell, f);

          // geometry props
          double dS = cell->face(f)->measure();  // face area
          neighbor_cell = cell->neighbor(f);
          normal = fe_face_values.normal_vector(0); // 0 is gauss point
          dx_ij = neighbor_cell->center() - cell->center();

          // neighbor cell data
          // const double poro_j = data.get_porosity->value(neighbor_cell->center());
          const double mu_j = data.get_viscosity();
          const double volume_factor_j = data.get_volume_factor();

          // get absolute perm
          for (int d=0; d<dim; d++)
            perm_j[d] = data.get_permeability->value(neighbor_cell->center(), d);
          // std::cout << perm_j[0] << "\t" << perm_j[1]<< "\t" << std::endl;

          // get relative permeability!!!!!!!!!!!!!!

          // Face properties
          harmonic_mean(perm_i, perm_j, perm_ij);
          const double mu_ij = arithmetic_mean(mu_i, mu_j);
          // const double poro_ij = arithmetic_mean(poro_i, poro_j);
          const double volume_factor_ij = arithmetic_mean(volume_factor_i,
                                                          volume_factor_j);
          // upwind relative permeability!!!!!!!!!!!!!!

          // Face transmissibility
          double T_ij = get_transmissibility(perm_ij, mu_ij, volume_factor_ij,
                                             normal, dx_ij, dS);
          // std::cout << "trans: " << T_ij << std::endl;

          matrix_ii += T_ij;
          system_matrix.add(i, j, -T_ij);
        } // end face loop
        else  // if the neighbor doesn't exist
          continue;

      }  // end face loop
      system_matrix.add(i, i, matrix_ii);
      rhs_vector[i] += rhs_i;

      // std::cout << "------------------------------\n";
    } // end cell loop

  } // eom


  template <int dim>
  void PressureSolver<dim>::print_system_matrix()
  {
    // for (unsigned int i=0; i<dof_handler.n_dofs(); i++) {
    //   for (unsigned int j=0; j<dof_handler.n_dofs(); j++) {
    //     try {
    //       std::cout << system_matrix(i, j) << "\t";
    //     }
    //     catch (...)
    //     {
    //       std::cout << 0 << "\t";
    //     }
    //   }
    //   std::cout << std::endl;
    // }

    // out, precision, scientific
    system_matrix.print_formatted(std::cout, 1, false);
  } // eom
}  // end of namespace

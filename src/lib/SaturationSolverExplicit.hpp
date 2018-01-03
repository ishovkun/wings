#pragma once

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/dofs/dof_handler.h>


namespace FluidSolvers
{
  template <int dim>
  class SaturationSolverExplicit
  {
  public:
    SaturationSolverExplicit(const int                n_phases_,
                             MPI_Comm                 &mpi_communicator_,
                             const DoFHandler<dim>    &dof_handler_,
                             const Model::Model<dim>  &model_,
                             ConditionalOStream       &pcout_);
    void setup_dofs(IndexSet &locally_owned_dofs,
                    IndexSet &locally_relevant_dofs);
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
  SaturationSolverExplicit<dim>::
  SaturationSolverExplicit(const int                  n_phases_,
                           MPI_Comm                   &mpi_communicator_,
                           const DoFHandler<dim>      &dof_handler_,
                           const Model::Model<dim>    &model_,
                           ConditionalOStream         &pcout_)
    :
    n_phases(n_phases_),
    mpi_communicator(mpi_communicator_),
    dof_handler(dof_handler_),
    model(model_),
    pcout(pcout_)
  {}


  template <int dim>
  void
  SaturationSolverExplicit<dim>::setup_dofs(IndexSet &locally_owned_dofs,
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
} // end of namespace

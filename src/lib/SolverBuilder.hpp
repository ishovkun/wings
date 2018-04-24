#pragma once
#include <deal.II/distributed/tria.h>

#include <Model.hpp>
#include <Probe.hpp>

#include <SolverIMPES.hpp>
#include <ElasticSolver.hpp>

#include <Equations/IMPESPressure.hpp>
#include <Equations/IMPESSaturation.hpp>


namespace Wings
{
using namespace dealii;


template<int dim, int n_phases>
class SolverBuilder
{
 public:
  SolverBuilder(const Model::Model<dim>                   & model,
                Probe::Probe<dim,n_phases>                & probe,
                MPI_Comm                                  & mpi_communicator,
                parallel::distributed::Triangulation<dim> & triangulation,
                ConditionalOStream                        & pcout);

  void build_solvers();

  std::shared_ptr<FluidSolvers::FluidSolverBase<dim,n_phases>> get_fluid_solver();
  std::shared_ptr<SolidSolvers::SolidSolverBase<dim,n_phases>> get_solid_solver();

 protected:
  void build_fluid_solver();
  void build_solid_solver();
  void couple_solvers();

  const Model::Model<dim>                   & model;
  Probe::Probe<dim,n_phases>                & probe;
  MPI_Comm                                  & mpi_communicator;
  parallel::distributed::Triangulation<dim> & triangulation;
  ConditionalOStream                        & pcout;

  std::shared_ptr<FluidSolvers::FluidSolverBase<dim,n_phases>> fluid_solver;
  std::shared_ptr<SolidSolvers::SolidSolverBase<dim,n_phases>> solid_solver;

};



template<int dim, int n_phases>
SolverBuilder<dim,n_phases>::
SolverBuilder(const Model::Model<dim>                   & model,
              Probe::Probe<dim,n_phases>                & probe,
              MPI_Comm                                  & mpi_communicator,
              parallel::distributed::Triangulation<dim> & triangulation,
              ConditionalOStream                        & pcout)
    :
    model(model),
    probe(probe),
    mpi_communicator(mpi_communicator),
    triangulation(triangulation),
    pcout(pcout)
{}



template<int dim, int n_phases>
void SolverBuilder<dim,n_phases>::build_fluid_solver()
{
  Equations::IMPESPressure<n_phases> implicit_pressure(model, probe);
  Equations::IMPESSaturation<n_phases> explicit_saturation(model, probe);
  fluid_solver =
      std::make_shared<FluidSolvers::SolverIMPES<dim,n_phases>>
            (mpi_communicator,
             triangulation,
             model,
             pcout,
             implicit_pressure,
             explicit_saturation);
} // eom



template<int dim, int n_phases>
void SolverBuilder<dim,n_phases>::couple_solvers()
{
  if (model.solid_model != Model::SolidModelType::Compressibility)
  {
      const FEValuesExtractors::Vector displacement(0);
      // solid_solver->set_coupling(fluid_solver->get_dof_handler());
      // fluid_solver->set_coupling(solid_solver->get_dof_handler(),
      //                            solid_solver->relevant_solution,
      //                            solid_solver->old_solution,
      //                            displacement);

  }
} // eom



template<int dim, int n_phases>
void SolverBuilder<dim,n_phases>::build_solid_solver()
{
  // build solid solver
  switch(model.solid_model)
  {

    case Model::SolidModelType::Compressibility:
      {
        // maybe some dummy solver
        break;
      }

    case Model::SolidModelType::Elasticity:
      {
        solid_solver =
            std::make_shared<SolidSolvers::ElasticSolver<dim,n_phases>>
            (mpi_communicator, triangulation, model, pcout);
        break;
      }

    default:
      {
        throw(ExcMessage("solid solver undefined"));
        break;
      }
  } // end switch
} // eom



template<int dim, int n_phases>
void SolverBuilder<dim,n_phases>::build_solvers()
{

  build_fluid_solver();
  build_solid_solver();
  couple_solvers();
}  // end build_solvers



template<int dim, int n_phases>
std::shared_ptr<FluidSolvers::FluidSolverBase<dim,n_phases>>
SolverBuilder<dim,n_phases>::get_fluid_solver()
{
  return fluid_solver;
}  // eom



template<int dim, int n_phases>
std::shared_ptr<SolidSolvers::SolidSolverBase<dim,n_phases>>
SolverBuilder<dim,n_phases>::get_solid_solver()
{
  return solid_solver;
}  // eom

} // end of namespace

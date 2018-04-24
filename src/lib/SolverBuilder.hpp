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


static const int dim = 3;


class SolverBuilder
{
 public:
  SolverBuilder(const Model::Model<dim>                   & model,
                Probe::Probe                              & probe,
                MPI_Comm                                  & mpi_communicator,
                parallel::distributed::Triangulation<dim> & triangulation,
                ConditionalOStream                        & pcout);

  void build_solvers();

  // FluidSolvers::FluidSolverBase<dim>
  std::shared_ptr<FluidSolvers::FluidSolverBase>   get_fluid_solver();
  std::shared_ptr<SolidSolvers::SolidSolverBase>   get_solid_solver();

 protected:
  void build_fluid_solver();
  void build_solid_solver();
  void couple_solvers();

  const Model::Model<dim> & model;
  MPI_Comm                                  & mpi_communicator;
  parallel::distributed::Triangulation<dim> & triangulation;
  ConditionalOStream                        & pcout;

  std::shared_ptr<FluidSolvers::FluidSolverBase>    fluid_solver;
  std::shared_ptr<SolidSolvers::ElasticSolver<dim>> solid_solver;

};



SolverBuilder::
SolverBuilder(const Model::Model<dim>                   & model,
              Probe::Probe                              & probe,
              MPI_Comm                                  & mpi_communicator,
              parallel::distributed::Triangulation<dim> & triangulation,
              ConditionalOStream                        & pcout)
    :
    model(model),
    mpi_communicator(mpi_communicator),
    triangulation(triangulation),
    pcout(pcout)
{}



void SolverBuilder::build_fluid_solver()
{
  switch(model.n_phases())
  {
    case 1:
      {
        Equations::IMPESPressure<1> implicit_pressure(model);
        Equations::IMPESSaturation<1> explicit_saturation(model);
        fluid_solver =
            std::make_shared<FluidSolvers::SolverIMPES<1>>
            (mpi_communicator,
             triangulation,
             model,
             pcout,
             implicit_pressure,
             explicit_saturation);
        break;
      }

    case 2:
      {
        throw(ExcNotImplemented());
        break;
      }

    case 3:
      {
        throw(ExcNotImplemented());
        break;
      }

    default:
      {
        throw(ExcMessage("fluid solver undefined"));
      }
  } // end switch
} // eom



void SolverBuilder::couple_solvers()
{
  if (model.solid_model != Model::SolidModelType::Compressibility)
  {
      const FEValuesExtractors::Vector displacement(0);
      solid_solver->set_coupling(fluid_solver->get_dof_handler());
      fluid_solver->set_coupling(solid_solver->get_dof_handler(),
                                 solid_solver->relevant_solution,
                                 solid_solver->old_solution,
                                 displacement);

  }
} // eom



void SolverBuilder::build_solid_solver()
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
            std::make_shared<SolidSolvers::ElasticSolver<dim>>
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



void SolverBuilder::build_solvers()
{

  build_fluid_solver();
  build_solid_solver();
  couple_solvers();
}  // end build_solvers



std::shared_ptr<FluidSolvers::FluidSolverBase>
SolverBuilder::get_fluid_solver()
{
  return fluid_solver;
}  // eom



std::shared_ptr<SolidSolvers::SolidSolverBase>
SolverBuilder::get_solid_solver()
{
  return solid_solver;
}  // eom

} // end of namespace

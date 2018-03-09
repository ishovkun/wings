#pragma once
#include <deal.II/distributed/tria.h>

#include <Model.hpp>

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
  SolverBuilder(const Model::Model<dim> &model);

  // FluidSolvers::FluidSolverBase<dim>
  std::unique_ptr<FluidSolvers::FluidSolverBase<dim>>
  get_fluid_solver(MPI_Comm                                  & mpi_communicator,
                   parallel::distributed::Triangulation<dim> & triangulation,
                   ConditionalOStream                        & pcout);
  std::unique_ptr<SolidSolvers::ElasticSolver<dim>> get_solid_solver();

 protected:
  const Model::Model<dim> & model;
};



SolverBuilder::SolverBuilder(const Model::Model<dim> &model)
    :
    model(model)
{}



// FluidSolvers::FluidSolverBase<dim>
std::unique_ptr<FluidSolvers::FluidSolverBase<dim>>
SolverBuilder::
get_fluid_solver(MPI_Comm                                  & mpi_communicator,
                 parallel::distributed::Triangulation<dim> & triangulation,
                 ConditionalOStream                        & pcout)
{
  switch(model.n_phases())
  {
    case 1:
      {
        Equations::IMPESPressure<1,dim> implicit_pressure(model);
        Equations::IMPESSaturation<1,dim> explicit_saturation(model);
        // return std::make_unique<FluidSolvers::SolverIMPES<1,dim>>
        //     (new FluidSolvers::SolverIMPES<1,dim>(mpi_communicator,
        //                                           triangulation,
        //                                           model,
        //                                           pcout,
        //                                           implicit_pressure,
        //                                           explicit_saturation));
      }

    // case 2:
    //   {
    //     FluidEquations::FluidEquationsPressure<dim,2>   implicit_pressure(model);
    //     FluidEquations::FluidEquationsSaturation<dim,2> explicit_saturation(model);
    //     return std::make_unique<FluidSolvers::SolverIMPES<dim,2>>
    //         (new SolverIMPES<dim,2>(mpi_communicator,
    //                                 triangulation,
    //                                 model,
    //                                 pcout,
    //                                 implicit_pressure, explicit_saturation));
    //   }
    // case 3:
    //   {
    //     FluidEquations::FluidEquationsPressure<dim,3>   implicit_pressure(model);
    //     FluidEquations::FluidEquationsSaturation<dim,3> explicit_saturation(model);
    //     return std::make_unique<FluidSolvers::SolverIMPES<dim,3>>
    //         (new SolverIMPES<dim,3>(mpi_communicator,
    //                                 triangulation,
    //                                 model,
    //                                 pcout,
    //                                 implicit_pressure, explicit_saturation));
    //   }

  } // end switch
}  // eom
} // end of namespace

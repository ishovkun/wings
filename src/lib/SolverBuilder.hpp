#pragma once

#include <Model.hpp>

#include <SolverIMPES.hpp>
#include <ElasticSolver.hpp>

#include <Equations/IMPESPressure.hpp>
#include <Equations/IMPESSaturation.hpp>


namespace Wings
{

class SolverBuilder
{
  static const int dim = 3;
 public:
  SolverBuiler(const Model::Model<dim> &model);

  template<int n_phases>
  std::unique_ptr<FluidSolvers::SolverIMPES<dim,n_phases>>
  get_fluid_solver(MPI_Comm                                  & mpi_communicator_,
                   parallel::distributed::Triangulation<dim> & triangulation_,
                   const Model::Model<dim>                   & model_,
                   ConditionalOStream                        & pcout_);
  std::unique_ptr<SolidSolvers::ElasticSolver<dim>> get_solid_solver();

 protected:
  const Model::Model<dim> & model;
};



SolverBuilder::SolverBuilder(const Model::Model<dim> &model)
    :
    model(model)
{}



template<int n_phases>
std::unique_ptr<FluidSolvers::SolverIMPES<dim>>
SolverBuilder::
get_fluid_solver(MPI_Comm                                  & mpi_communicator_,
                 parallel::distributed::Triangulation<dim> & triangulation_,
                 const Model::Model<dim>                   & model_,
                 ConditionalOStream                        & pcout_)
{
  switch(model.n_phases())
  {
    case 1:
      {
        FluidEquations::FluidEquationsPressure<dim,1> implicit_pressure(model);
        FluidEquations::FluidEquationsSaturation<dim,1> explicit_saturation(model);
        return std::make_unique<FluidSolvers::SolverIMPES<dim,1>>
            (new SolverIMPES<dim,1>(mpi_communicator,
                                    triangulation,
                                    model,
                                    pcout,
                                    implicit_pressure, explicit_saturation));
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

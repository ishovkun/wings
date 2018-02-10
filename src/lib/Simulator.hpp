#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/grid/grid_generator.h> // to create mesh
#include <deal.II/grid/grid_out.h>

// Custom modules
#include <Model.hpp>
#include <Reader.hpp>
#include <OutputHelper.hpp>

// #include <Wellbore.hpp>
#include <SolverIMPES.hpp>
// #include <PressureSolver.hpp>
// #include <SaturationSolver.hpp>
#include <ElasticSolver.hpp>
#include <FEFunction/FEFunction.hpp>
// #include <FEFunction/FEFunctionPVT.hpp>



namespace Wings
{
using namespace dealii;


template <int dim>
class Simulator
{
 public:
  Simulator(std::string);
  // ~Simulator();
  void read_mesh();
  void create_mesh();
  void run();


 private:
  void refine_mesh();
  // void field_report(const double time_step,
  //                   const unsigned int time_step_number,
  //                   const FluidSolvers::SaturationSolver<dim> &saturation_solver);

  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  ConditionalOStream                        pcout;
  Model::Model<dim>                         model;
  // FluidSolvers::PressureSolver<dim>         pressure_solver;
  std::string                               input_file;
  Output::OutputHelper<dim>                 output_helper;
  // TimerOutput                               computing_timer;
};


template <int dim>
Simulator<dim>::Simulator(std::string input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    model(mpi_communicator, pcout),
    // pressure_solver(mpi_communicator, triangulation, model, pcout),
    input_file(input_file_name_),
    output_helper(mpi_communicator, triangulation)
    // ,computing_timer(mpi_communicator, pcout,
    //                 TimerOutput::summary, TimerOutput::wall_times)
{}



template <int dim>
void Simulator<dim>::create_mesh()
{
  // make grid with 102x1x1 elements,
  // hx = hy = hz = h = 25 ft
  std::vector<unsigned int > repetitions = {102, 1, 1};
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            repetitions,
                                            Point<dim>(0,   -25, -12.5),
                                            Point<dim>(510, 25, +12.5));
  GridOut grid_out;
  // std::ofstream out ("bl-mesh.vtk ");
  // grid_out.write_vtk(triangulation, out);
  std::ofstream out("buckley_leverett.msh");
  grid_out.write_msh(triangulation, out);

  GridTools::scale(model.units.length(), triangulation);
}  // eom



template <int dim>
void Simulator<dim>::read_mesh()
{
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(model.mesh_file.string());

  pcout << "Reading mesh file " << model.mesh_file << std::endl;
  // typename GridIn<dim>::format format = gridin<dim>::ucd;
  // gridin.read(f, format);
  gridin.read_msh(f);
  GridTools::scale(model.units.length(), triangulation);
}  // eom



template <int dim>
void Simulator<dim>::refine_mesh()
{
  typename Triangulation<dim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();

  for (;cell != endc; ++cell)
    if (!cell->is_artificial())
    {
      if (
              abs(cell->center()[0] - 1.5) < DefaultValues::small_number
              &&
              abs(cell->center()[1] - 2.5) < DefaultValues::small_number
          )
      {
        cell->set_refine_flag();
        break;
      }
    }

  triangulation.prepare_coarsening_and_refinement();
  triangulation.execute_coarsening_and_refinement();
} // eom



// template <int dim>
// void
// Simulator<dim>::
// field_report(const double time,
//              const unsigned int time_step_number,
//              const FluidSolvers::SaturationSolver<dim> &saturation_solver)
// {
//   DataOut<dim> data_out;

//   data_out.attach_dof_handler(pressure_solver.get_dof_handler());
//   data_out.add_data_vector(pressure_solver.relevant_solution, "pressure",
//                            DataOut<dim>::type_dof_data);
//   data_out.add_data_vector(saturation_solver.relevant_solution[0], "Sw",
//                            DataOut<dim>::type_dof_data);
//   data_out.build_patches();

//   output_helper.write_output(time, time_step_number, data_out);

// }  // eom



template <int dim>
void Simulator<dim>::run()
{
  Parsers::Reader reader(pcout, model);
  reader.read_input(input_file, /* verbosity= */0);
  output_helper.set_case_name("solution");


  read_mesh();
  // create_mesh();

  output_helper.prepare_output_directories();

  FluidSolvers::SolverIMPES<dim>(mpi_communicator, triangulation, model, pcout);

  // FluidSolvers::SaturationSolver<dim>
  //     saturation_solver(mpi_communicator,
  //                       pressure_solver.get_dof_handler(),
  //                       model, pcout);

  // SolidSolvers::ElasticSolver<dim>
  //     solid_solver(mpi_communicator, triangulation, model, pcout);

  // pressure_solver.setup_dofs();

  // // if multiphase
  // saturation_solver.setup_dofs(pressure_solver.locally_owned_dofs,
  //                              pressure_solver.locally_relevant_dofs);


  // // initial values
  // for (unsigned int i=0; i<saturation_solver.solution[0].size(); ++i)
  // {
  //   saturation_solver.solution[0][i] =model.residual_saturation_water();
  //   pressure_solver.solution = 1000*model.units.pressure();
  //   // pressure_solver.solution = 1e8;
  // }
  // pressure_solver.relevant_solution = pressure_solver.solution;
  // saturation_solver.relevant_solution[0] = saturation_solver.solution[0];

  // model.locate_wells(pressure_solver.get_dof_handler());

  // CellValues::CellValuesBase<dim> cell_values_pressure(model),
  //                                 neighbor_values_pressure(model);
  // CellValues::CellValuesSaturation<dim> cell_values_saturation(model);

  // FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
  //     pressure_function(pressure_solver.get_dof_handler(),
  //                       pressure_solver.relevant_solution);
  // FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
  //     saturation_function(pressure_solver.get_dof_handler(),
  //                         saturation_solver.relevant_solution);

  // {
  //   Vector<double> sat(2);
  //   std::vector<double> rperm(2);
  //   sat[0] = 0.2;
  //   sat[1] = 0.8;
  //   model.get_relative_permeability(sat, rperm);
  //   AssertThrow(rperm[0] < DefaultValues::small_number,
  //               ExcMessage("bug in rel perm"));
  //   AssertThrow(Math::relative_difference(rperm[1], 0.8)  < DefaultValues::small_number,
  //               ExcMessage("bug in rel perm"));
  // }

  // double time = 0;
  // double time_step = model.min_time_step;
  // unsigned int time_step_number = 0;

  // while(time <= model.t_max)
  // {
  //   time += time_step;
  //   pressure_solver.old_solution = pressure_solver.solution;

  //   pcout << "time " << time << std::endl;
  //   model.update_well_controls(time);
  //   model.update_well_productivities(pressure_function, saturation_function);

  //   { // solve for pressure
  //     pressure_solver.assemble_system(cell_values_pressure, neighbor_values_pressure,
  //                                     time_step,
  //                                     saturation_solver.relevant_solution);
  //     pressure_solver.solve();
  //     pressure_solver.relevant_solution = pressure_solver.solution;
  //   }

  //   { // solve for saturation
  //     saturation_solver.solve(cell_values_saturation,
  //                             neighbor_values_pressure,
  //                             time_step,
  //                             pressure_solver.relevant_solution,
  //                             pressure_solver.old_solution);
  //     saturation_solver.relevant_solution[0] = saturation_solver.solution[0];
  //     saturation_solver.relevant_solution[1] = saturation_solver.solution[1];
  //   }


  //   field_report(time, time_step_number, saturation_solver);

  //   time_step_number++;
  // } // end time loop

} // eom


} // end of namespace

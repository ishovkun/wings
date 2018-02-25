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
  void read_mesh(unsigned int verbosity = 0);
  void create_mesh();
  void run();


 private:
  void refine_mesh();
  // export vtu data (for paraview)
  void field_report(const double                           time_step,
                    const unsigned int                     time_step_number,
                    const FluidSolvers::SolverIMPES<dim> & fluid_solver);
  // Solve time step for a blackoil system without geomechanics
  void solve_time_step_fluid(FluidSolvers::SolverIMPES<dim> & fluid_solver,
                             const double                     time_step);
  // Solve time step for a blackoil system with geomechanics
  void solve_time_step_fluid_mechanics(FluidSolvers::SolverIMPES<dim>   & fluid_solver,
                                       SolidSolvers::ElasticSolver<dim> & solid_solver,
                                       const double                       time_step);

  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  ConditionalOStream                        pcout;
  Model::Model<dim>                         model;
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
  const auto & p1 = model.mesh_config.points.first;
  const auto & p2 = model.mesh_config.points.second;

  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            model.mesh_config.n_cells,
                                            p1, p2);

  // make boundary ids
  typename Triangulation<3>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();

  for (; cell!=endc; ++cell)
    for (unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f)
    {
      const Point<dim> face_center = cell->face(f)->center();
      if (cell->face(f)->at_boundary())
      {
        // left
        if (abs(face_center[0] - p1[0]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(0);
        }
        // right
        if (abs(face_center[0] - p2[0]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(1);
        }
        // front
        if (abs(face_center[1] - p1[1]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(2);
        }
        // back
        if (abs(face_center[1] - p2[1]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(3);
          // std::cout << "back" << std::endl;
        }
        // bottom
        if (abs(face_center[2] - p1[2]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(4);
        }
        // top
        if (abs(face_center[2] - p2[2]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(5);
        }
      }  // end if face at boundary

    }  // end cell and face loop

  GridOutFlags::Msh flags(/* write_faces = */ true,
                          /* write_lines = */ false);
  GridOut grid_out;
  grid_out.set_flags(flags);
  // std::ofstream out ("bl-mesh.vtk ");
  // grid_out.write_vtk(triangulation, out);
  std::ofstream out(input_file + ".msh");
  grid_out.write_msh(triangulation, out);

  GridTools::scale(model.units.length(), triangulation);
}  // eom



template <int dim>
void Simulator<dim>::read_mesh(unsigned int verbosity)
{
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(model.mesh_config.file.string());

  if (verbosity > 0)
    pcout << "Reading mesh file " << model.mesh_config.file
          << std::flush;

  if (model.mesh_config.type == Model::MeshType::Msh)
    gridin.read_msh(f);
  else if (model.mesh_config.type == Model::MeshType::Abaqus)
    gridin.read_abaqus(f);
  pcout << " OK" << std::endl;
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



template <int dim>
void
Simulator<dim>::
field_report(const double time,
             const unsigned int time_step_number,
             const FluidSolvers::SolverIMPES<dim> &fluid_solver)
{
  DataOut<dim> data_out;

  // data_out.attach_dof_handler(pressure_solver.get_dof_handler());
  // data_out.add_data_vector(pressure_solver.relevant_solution, "pressure",
  //                          DataOut<dim>::type_dof_data);
  // data_out.add_data_vector(saturation_solver.relevant_solution[0], "Sw",
  //                          DataOut<dim>::type_dof_data);
  fluid_solver.attach_data(data_out);
  data_out.build_patches();

  output_helper.write_output(time, time_step_number, data_out);

}  // eom


template<int dim>
void
Simulator<dim>::solve_time_step_fluid(FluidSolvers::SolverIMPES<dim> & fluid_solver,
                                      const double                     time_step)
{
  // update wells
  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      pressure_function(fluid_solver.get_dof_handler(),
                        fluid_solver.pressure_relevant);
  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      saturation_function(fluid_solver.get_dof_handler(),
                          fluid_solver.saturation_relevant);

  model.update_well_productivities(pressure_function, saturation_function);

  // solve for fluid flow
  fluid_solver.assemble_pressure_system(time_step);
  fluid_solver.solve_pressure_system();
  fluid_solver.pressure_relevant = fluid_solver.solution;
  if (model.n_phases() > 1)
  {
    fluid_solver.solve_saturation_system(time_step);
    fluid_solver.saturation_relevant[0] = fluid_solver.solution;
  }
}  // end solve_time_step_fluid


template<int dim>
void
Simulator<dim>::
solve_time_step_fluid_mechanics(FluidSolvers::SolverIMPES<dim>   & fluid_solver,
                                SolidSolvers::ElasticSolver<dim> & solid_solver,
                                const double                       time_step)
{
  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      pressure_function(fluid_solver.get_dof_handler(),
                        fluid_solver.pressure_relevant);
  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      saturation_function(fluid_solver.get_dof_handler(),
                          fluid_solver.saturation_relevant);
  TrilinosWrappers::MPI::Vector pressure_old_iter(fluid_solver.pressure_relevant);

  int fss_step = 0;
  while(fss_step < model.max_coupling_steps)
  {
    fss_step++;
    pcout << "fss step " << fss_step << " of " << model.max_coupling_steps
          << std::flush;

    // model.update_well_productivities(pressure_function, saturation_function);
    pressure_old_iter = fluid_solver.pressure_relevant;

    {// solve for fluid flow
      fluid_solver.assemble_pressure_system(time_step);
      fluid_solver.solve_pressure_system();
      fluid_solver.pressure_relevant = fluid_solver.solution;
      if (model.n_phases() > 1)
      {
        fluid_solver.solve_saturation_system(time_step);
        fluid_solver.saturation_relevant[0] = fluid_solver.solution;
      }
    } // end solve fluid flow
    { // solve elasticity
      solid_solver.assemble_system(fluid_solver.pressure_relevant);
      solid_solver.solve();
      solid_solver.relevant_solution = solid_solver.solution;
    }

    // estimate error
    double error = 0;
    for (const auto & dof : fluid_solver.locally_relevant_dofs)
    {
      const double diff =
          fluid_solver.pressure_relevant[dof] - pressure_old_iter[dof];
      error += diff*diff;
    }
    error = Utilities::MPI::sum(error, mpi_communicator);
    error /= abs(fluid_solver.pressure_relevant.mean_value());
    pcout << "\t" << error << std::endl;

    if (error < model.coupling_tolerance)
      return;
  } // end fss loop

  AssertThrow(false, ExcMessage("FSS didn't converge"));
}  // end solve_



template <int dim>
void Simulator<dim>::run()
{
  Parsers::Reader reader(pcout, model);
  reader.read_input(input_file, /* verbosity= */0);
  output_helper.set_case_name("solution");


  if (model.mesh_config.type == Model::MeshType::Create)
    create_mesh();
  else
    read_mesh(/* verbosity = */ 1);

  // output_helper.prepare_output_directories();

  // define fluid object
  CellValues::CellValuesPressure<dim> cell_values(model),
                                      neighbor_values(model);
  CellValues::CellValuesSaturation<dim> cell_values_saturation(model);
  FluidSolvers::SolverIMPES<dim> fluid_solver(mpi_communicator,
                                              triangulation,
                                              model, pcout,
                                              cell_values, neighbor_values,
                                              cell_values_saturation);

  // define solid object
  SolidSolvers::ElasticSolver<dim>
      solid_solver(mpi_communicator, triangulation, model, pcout);

  // pcout << "solid definition sucess" << std::endl;

  // couple solvers
  if (model.solid_model != Model::SolidModelType::Compressibility)
  {
    const FEValuesExtractors::Vector displacement(0);
    solid_solver.set_coupling(fluid_solver.get_dof_handler());
    fluid_solver.set_coupling(solid_solver.get_dof_handler(),
                              solid_solver.relevant_solution,
                              solid_solver.old_solution,
                              displacement);
    solid_solver.setup_dofs();
  }

  fluid_solver.setup_dofs();

  model.locate_wells(fluid_solver.get_dof_handler());

  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      pressure_function(fluid_solver.get_dof_handler(),
                        fluid_solver.pressure_relevant);
  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      saturation_function(fluid_solver.get_dof_handler(),
                          fluid_solver.saturation_relevant);

  { // fluid initialization step
    fluid_solver.pressure_relevant = model.reference_pressure;
    if (model.n_phases() == 2)
      fluid_solver.saturation_relevant[0] = model.initial_saturation_water;
  }

  if (model.solid_model != Model::SolidModelType::Compressibility)
  { // geomechanics initialization step
    // solid.solver.initialize(fluid_solver.pressure_relevant);
    solid_solver.assemble_system(fluid_solver.pressure_relevant);
    solid_solver.solve();
    solid_solver.relevant_solution = solid_solver.solution;
  }

  double time = 0;
  int time_step_number = 0;
  while (time < model.t_max)
  {
    double time_step = model.min_time_step;
    time += time_step;
    time_step_number++;
    pcout << "Time " << time << "; time step " << time_step << std::endl;

    fluid_solver.pressure_old = fluid_solver.pressure_relevant;
    model.update_well_controls(time);
    // try
    // {
    if (model.solid_model == Model::SolidModelType::Compressibility)
      solve_time_step_fluid(fluid_solver, time_step);
    else if (model.solid_model == Model::SolidModelType::Elasticity)
      solve_time_step_fluid_mechanics(fluid_solver, solid_solver, time_step);
    // }
    // catch (...)
    // {
    //   // truncate time step
    //   AssertThrow(false, ExcNotImplemented());
    // }

    // field_report(time, time_step_number, fluid_solver);

  } // end time loop

  // solid_solver.solution.print(std::cout, 4, true, false);
  // fluid_solver.initialize(cell_values, neighbor_values, time_step);
  // fluid_solver.assemble_pressure_system(time_step);


} // eom


} // end of namespace

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
  // std::vector<unsigned int > repetitions = {3, 3, 1};
  // GridGenerator::subdivided_hyper_rectangle(triangulation,
  //                                           repetitions,
  //                                           Point<dim>(0, 0, -0.5),
  //                                           Point<dim>(3, 3, 0.5));
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
          // std::cout << "left " << cell->center() << "\t" << f << std::endl;
        }
        // right
        if (abs(face_center[0] - p2[0]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(1);
          // std::cout << "right " << cell->center() << "\t" << f << std::endl;
        }
        // front
        if (abs(face_center[1] - p1[1]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(2);
          // std::cout << "front" << std::endl;
          // std::cout << "front " << cell->center() << "\t" << f << std::endl;
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
          // std::cout << "bottom" << std::endl;
        }
        // top
        if (abs(face_center[2] - p2[2]) < DefaultValues::small_number_geometry)
        {
          cell->face(f)->set_boundary_id(5);
          // std::cout << "top" << std::endl;
        }
      }  // end if face at boundary

    }  // end cell and face loop

  GridOutFlags::Msh flags(/* write_faces = */ true,
                          /* write_lines = */ false);
  GridOut grid_out;
  grid_out.set_flags(flags);
  // std::ofstream out ("bl-mesh.vtk ");
  // grid_out.write_vtk(triangulation, out);
  std::ofstream out("3x3x1_1m3.msh");
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
    pcout << "Reading mesh file " << model.mesh_config.file << std::endl;

  if (model.mesh_config.type == Model::MeshType::Msh)
    gridin.read_msh(f);
  else if (model.mesh_config.type == Model::MeshType::Abaqus)
    gridin.read_abaqus(f);

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


  if (model.mesh_config.type == Model::MeshType::Create)
    create_mesh();
  else
    read_mesh(/* verbosity = */ 1);

  // output_helper.prepare_output_directories();

  // create fluid and solid solver objects
  FluidSolvers::SolverIMPES<dim> fluid_solver(mpi_communicator,
                                              triangulation,
                                              model, pcout);

  SolidSolvers::ElasticSolver<dim>
      solid_solver(mpi_communicator, triangulation, model, pcout);

  // couple solvers
  const FEValuesExtractors::Vector displacement(0);
  solid_solver.set_coupling(fluid_solver.get_dof_handler());
  fluid_solver.set_coupling(solid_solver.get_dof_handler(),
                            solid_solver.relevant_solution,
                            solid_solver.old_solution,
                            displacement);

  fluid_solver.setup_dofs();
  solid_solver.setup_dofs();

  // fluid_solver.solution = 0.2;
  // fluid_solver.saturation_relevant[0] = fluid_solver.solution;
  // fluid_solver.saturation_relevant[1] = 1.0;
  // fluid_solver.saturation_relevant[1] -= fluid_solver.solution;

  // initial values
  // fluid_solver.solution = 1e6;
  // fluid_solver.pressure_relevant = fluid_solver.solution;

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

  // CellValues::CellValuesBase<dim>
  //     cell_values(model), neighbor_values(model);

  // // double time = 0;
  // double time_step = model.min_time_step;

  { // geomechanics initialization step
    solid_solver.assemble_system(fluid_solver.pressure_relevant);
    solid_solver.solve();
    solid_solver.relevant_solution = solid_solver.solution;
  }

  solid_solver.solution.print(std::cout, 4, true, false);

  // now we need to check if the strains are correct
  const double E = model.get_young_modulus->value(Point<3>(0,0,0));
  const double nu = model.get_poisson_ratio->value(Point<3>(0,0,0));
  const double sigma_v = -model.solid_neumann_values[0];
  const double sigma_h = sigma_v/E * (1.0 - 2*nu*nu/(1.0 - nu));

  // fluid_solver.assemble_pressure_system(cell_values, neighbor_values, time_step);
  // const auto & system_matrix = fluid_solver.get_system_matrix();
  // system_matrix.print(std::cout, true);
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

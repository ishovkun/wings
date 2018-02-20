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
  void field_report(const double                           time_step,
                    const unsigned int                     time_step_number,
                    const FluidSolvers::SolverIMPES<dim> & fluid_solver);

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

  model.locate_wells(fluid_solver.get_dof_handler());

  CellValues::CellValuesBase<dim> cell_values(model),
                                  neighbor_values(model);
  CellValues::CellValuesSaturation<dim> cell_values_saturation(model);

  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      pressure_function(fluid_solver.get_dof_handler(),
                        fluid_solver.pressure_relevant);
  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      saturation_function(fluid_solver.get_dof_handler(),
                          fluid_solver.saturation_relevant);

  // // double time = 0;
  double time_step = model.min_time_step;

  { // geomechanics initialization step
    solid_solver.assemble_system(fluid_solver.pressure_relevant);
    solid_solver.solve();
    solid_solver.relevant_solution = solid_solver.solution;
  }

  // solid_solver.solution.print(std::cout, 4, true, false);

  // now we need to check if the strains are correct
  const double E = model.get_young_modulus->value(Point<3>(0,0,0));
  const double nu = model.get_poisson_ratio->value(Point<3>(0,0,0));
  const double bulk_modulus = E/3.0/(1.0-2.0*nu);
  const double sigma_v = -model.solid_neumann_values[0];
  const double eps_v = -( sigma_v/E * (1.0 - 2*nu*nu/(1.0 - nu)) );

  const double tol = DefaultValues::small_number;
  // pcout << "eps_v" << eps_v << std::endl;
  // test vertical strain
  for (unsigned int i=0; i<solid_solver.solution.size(); ++i)
    // check only non-zero components
    // those that are vertical and not at constrained boundaries
    if (abs(solid_solver.solution[i]) > DefaultValues::small_number)
      if (Math::relative_difference(solid_solver.solution[i],
                                    eps_v) > DefaultValues::small_number)
      {
        pcout << "num[" << i << "] = " << solid_solver.solution[i] << std::endl;
        pcout << "an[" << i << "] = " << eps_v << std::endl;
        AssertThrow(false, ExcMessage("num["+std::to_string(i)+"] is wrong"));
      }


  fluid_solver.assemble_pressure_system(cell_values,
                                        neighbor_values, time_step);

  const auto & rhs_vector = fluid_solver.get_rhs_vector();
  // rhs_vector.print(std::cout, 3, true, false);

  // Since we have no wellbores and initial pressure is 0,
  // the pressure rhs term only should contain the poroelastic term
  // - alpha/B_fluid d e_v / dt
  // Since fluid formulation uses piecewise constants, volumetric strain
  // in each dof is the same
  // We constrained all horizontal strains, so the volumetric strain is equal
  // to the vertical strain
  const double alpha = model.get_biot_coefficient();
  double rhs_an = -alpha*eps_v/time_step;
  for (unsigned int i=0; i<rhs_vector.size(); ++i)
    if (Math::relative_difference(rhs_vector[i],
                                  rhs_an) > tol)
    {
      pcout << "num[" << i << "] = " << rhs_vector[i] << std::endl;
      pcout << "an[" << i << "] = " << rhs_an << std::endl;
      AssertThrow(false, ExcMessage("num["+std::to_string(i)+"] is wrong"));
    }

  // diagonal of the system matrix should have some new terms
  const auto & system_matrix = fluid_solver.get_system_matrix();

  const double k = model.get_permeability->value(Point<dim>(1,1,1), 1);
  // pcout << "perm = " << k << std::endl;
  const double mu = 1e-3;
  const double B_w = 1;
  const double cw = 5e-10;
  const double h = 1;
  const double phi = model.get_porosity->value(Point<dim>(1,1,1), 1);
  // // Compute transmissibility and mass matrix entries
  const double T = 1./mu/B_w*(k/h)*h*h;
  const double B = h*h*h/B_w*(phi*cw + alpha*alpha/bulk_modulus);
  //   field_report(time, time_step_number, saturation_solver);

  // // Testing A(0, 0) - two neighbors
  int dof = 0;
  double num = system_matrix(dof, dof);
  double an = B/time_step + 2*T;
  // if (Math::relative_difference(num, an) > tol)
  // {
    pcout << "num[" << dof << ", " << dof << "] " << num << std::endl;
    pcout << "an[" << dof << ", " << dof << "] " << an << std::endl;
    pcout << "rdiff " << Math::relative_difference(num, an) << std::endl;
  // }
  //   time_step_number++;
  // } // end time loop

} // eom


} // end of namespace

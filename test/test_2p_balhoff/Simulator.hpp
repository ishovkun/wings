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

// #include <Wellbore.hpp>
#include <PressureSolver.hpp>
#include <SaturationSolver.hpp>
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
  void
  compare_with_analytics(FluidSolvers::SaturationSolver<dim> &saturation_solver);


 private:
  void refine_mesh();
  void field_report(const double time_step,
                    const unsigned int time_step_number,
                    const FluidSolvers::SaturationSolver<dim> &saturation_solver);

  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  ConditionalOStream                        pcout;
  Model::Model<dim>                         model;
  FluidSolvers::PressureSolver<dim>         pressure_solver;
  std::string                               input_file;
  // TimerOutput                               computing_timer;
};


template <int dim>
Simulator<dim>::Simulator(std::string input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    model(mpi_communicator, pcout),
    pressure_solver(mpi_communicator, triangulation, model, pcout),
    input_file(input_file_name_)
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



template <int dim>
void
Simulator<dim>::
field_report(const double time,
             const unsigned int time_step_number,
             const FluidSolvers::SaturationSolver<dim> &saturation_solver)
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(pressure_solver.get_dof_handler());
  data_out.add_data_vector(pressure_solver.relevant_solution, "pressure",
                           DataOut<dim>::type_dof_data);
  data_out.add_data_vector(saturation_solver.relevant_solution[0], "Sw",
                           DataOut<dim>::type_dof_data);
  data_out.build_patches();

  std::ostringstream filename;
  // filename << "./" << data.output_directory
  //          << "/"  << data.data_set_name
  //          << Utilities::int_to_string(time_step_number, 4)
  //          << ".vtk";
  filename << "solution/solution."  << time_step_number << ".vtk";
  std::ofstream output(filename.str().c_str());
  data_out.write_vtk(output);
}




template <int dim>
void Simulator<dim>::run()
{
  Parsers::Reader reader(pcout, model);
  reader.read_input(input_file, /* verbosity= */0);

  // read_mesh();
  create_mesh();

  FluidSolvers::SaturationSolver<dim>
      saturation_solver(mpi_communicator,
                        pressure_solver.get_dof_handler(),
                        model, pcout);

  pressure_solver.setup_dofs();

  // tolerance for parameter check
  const double tol_p  = DefaultValues::small_number;
  { // test permeability function
    // first and last points have 1000 md perm, others - 50 md
    const double md = model.units.permeability();
    const double ft = model.units.length();
    const double k1 = model.get_permeability->value(Point<dim>(0.0, 0.0, 0.0));
    AssertThrow(Math::relative_difference(k1, 1000*md) < tol_p,
                ExcMessage("bug in relperm function"));
    // std::cout << k1/md << std::endl;
    const double k2 = model.get_permeability->value(Point<dim>(6*ft, 0.0, 0.0));
    AssertThrow(Math::relative_difference(k2, 50*md) < tol_p,
                ExcMessage("bug in relperm function"));
    const double k4 = model.get_permeability->value(Point<dim>(500*ft, 0.0, 0.0));
    AssertThrow(Math::relative_difference(k4, 50*md) < tol_p,
                ExcMessage("bug in relperm function"));
    // std::cout << k2/md << std::endl;
    const double k3 = model.get_permeability->value(Point<dim>(510*ft, 0.0, 0.0));
    AssertThrow(Math::relative_difference(k3, 1000*md) < tol_p,
                ExcMessage("bug in relperm function"));
    // std::cout << k3/md << std::endl;
  }

  // if multiphase
  saturation_solver.setup_dofs(pressure_solver.locally_owned_dofs,
                               pressure_solver.locally_relevant_dofs);


  // initial values
  for (unsigned int i=0; i<saturation_solver.solution[0].size(); ++i)
  {
    saturation_solver.solution[0][i] =model.residual_saturation_water();
    pressure_solver.solution = 1000*model.units.pressure();
    // pressure_solver.solution = 1e8;
  }
  pressure_solver.relevant_solution = pressure_solver.solution;
  saturation_solver.relevant_solution[0] = saturation_solver.solution[0];

  model.locate_wells(pressure_solver.get_dof_handler());

  CellValues::CellValuesBase<dim> cell_values_pressure(model),
                                  neighbor_values_pressure(model);
  CellValues::CellValuesSaturation<dim> cell_values_saturation(model);

  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      pressure_function(pressure_solver.get_dof_handler(),
                        pressure_solver.relevant_solution);
  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      saturation_function(pressure_solver.get_dof_handler(),
                          saturation_solver.relevant_solution);

  {
    Vector<double> sat(2);
    std::vector<double> rperm(2);
    sat[0] = 0.2;
    sat[1] = 0.8;
    model.get_relative_permeability(sat, rperm);
    pcout << "krw = " << rperm[0] << "\t"
          << "kro = " << rperm[1] << std::endl;
    // return;
  }

  double time = 0;
  double time_step = model.min_time_step;
  unsigned int time_step_number = 0;

  while(time <= model.t_max)
  {
    time += time_step;
    pressure_solver.old_solution = pressure_solver.solution;

    // pcout << "time " << time/model.units.time() << std::endl;
    // pcout << "tmax " << model.t_max/model.units.time() << std::endl;
    model.update_well_controls(time);
    model.update_well_productivities(pressure_function, saturation_function);

    { // solve for pressure
      pressure_solver.assemble_system(cell_values_pressure, neighbor_values_pressure,
                                      time_step,
                                      saturation_solver.relevant_solution);
      pressure_solver.solve();
      pressure_solver.relevant_solution = pressure_solver.solution;
    }

    { // solve for saturation
      saturation_solver.solve(cell_values_saturation,
                              neighbor_values_pressure,
                              time_step,
                              pressure_solver.relevant_solution,
                              pressure_solver.old_solution);
      saturation_solver.relevant_solution[0] = saturation_solver.solution[0];
      saturation_solver.relevant_solution[1] = saturation_solver.solution[1];
    }


    field_report(time, time_step_number, saturation_solver);

    time_step_number++;
    // if (time_step_number == 2) return;
  } // end time loop


  compare_with_analytics(saturation_solver);

  {  // construct dimensionless parameters and get saturation profile
    auto & injector = model.wells[0];
    const double q_rate = injector.get_control().value;
    const double ft = model.units.length();
    const double area = 25*50*ft*ft;
    const double length = 510*ft;
    // const auto & cell = pressure_solver
    //     .get_dof_handler()
    //     .begin_active();
    // const auto volume = cell->measure()*triangulation.n_cells();
    // pcout << "V = " << volume << std::endl;
    // pcout << "V1 = " << area*length << std::endl;
    const double phi = model.get_porosity->value(Point<dim>(0,0,0));
    const double dimensionless_time =
        q_rate*time/area/length/phi;
    pcout << "td = " << dimensionless_time << std::endl;
  }

} // eom



template<int dim>
void
Simulator<dim>::
compare_with_analytics(FluidSolvers::SaturationSolver<dim> &saturation_solver)
{
  std::string fname = "../test/test_buckley/analytical.txt";
  std::ifstream f(fname.c_str());
  std::stringstream buffer;
  buffer << f.rdbuf();
  std::string input_text = buffer.str();
  std::vector<double> analytical =
      Parsers::parse_string_list<double>(input_text, "\n");

  // std::cout << "size = " << analytical.size() <<std::endl;
  for (unsigned int i=1; i<saturation_solver.solution[0].size(); ++i)
    pcout << analytical[i-1] << "\t"
          << saturation_solver.solution[0][i]
          << std::endl;
}  // end compare_with_analytics

} // end of namespace

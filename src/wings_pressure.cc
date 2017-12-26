/*
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/distributed/tria.h>

// Custom modules
#include <Wellbore.hpp>
#include <PressureSolver.hpp>
#include <Parsers.hpp>
#include <CellValues.hpp>

namespace Wings
{
  using namespace dealii;


  template <int dim>
  class WingsPressure
  {
  public:
    WingsPressure(std::string);
    // ~WingsPressure();
    void read_mesh();
    void run();

  private:
    void refine_mesh();

    MPI_Comm                                  mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;
    FluidSolvers::PressureSolver<dim>         pressure_solver;
    Data::DataBase<dim>                       data;
    std::string                               input_file;

  };


  template <int dim>
  WingsPressure<dim>::WingsPressure(std::string input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator),
    pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    computing_timer(mpi_communicator, pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times),
    pressure_solver(mpi_communicator, triangulation, data, pcout),
    input_file(input_file_name_)
  {}


  template <int dim>
  void WingsPressure<dim>::read_mesh()
  {
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(data.mesh_file.string());

    // typename GridIn<dim>::format format = gridin<dim>::ucd;
    // gridin.read(f, format);
    gridin.read_msh(f);
  }  // eom


  template <int dim>
  void WingsPressure<dim>::refine_mesh()
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
  void WingsPressure<dim>::run()
  {
    data.read_input(input_file, /* verbosity = */ 0);
    // data.read_input(input_file, /* verbosity = */ 1);
    // data.print_input();
    // std::cout << "reading mesh file " << data.mesh_file.string() << std::endl;
    read_mesh();

    refine_mesh();
    // return;
    pressure_solver.setup_dofs();

    const auto & pressure_dof_handler = pressure_solver.get_dof_handler();
    const auto & pressure_fe = pressure_solver.get_fe();
    data.locate_wells(pressure_dof_handler, pressure_fe);
    data.update_well_productivities();

    for (auto & id : data.get_well_ids())
    {
      std::cout << "well_id " << id << std::endl;
      auto & well = data.wells[id];

      std::cout << "Real locations"  << std::endl;
      for (auto & loc : well.get_locations())
        std::cout << loc << std::endl;

      std::cout << "Assigned locations"  << std::endl;
      for (auto & cell : well.get_cells())
        std::cout << cell->center() << std::endl;

      std::cout << std::endl;
    }

    double time = 0;
    double time_step = data.get_time_step(time);
    // some init values
    pressure_solver.solution = 0;
    pressure_solver.solution[0] = 1;
    pressure_solver.old_solution = pressure_solver.solution;

    data.update_well_controls(time);

    CellValues::CellValuesBase<dim>
      cell_values(data), neighbor_values(data);
    pressure_solver.assemble_system(cell_values, neighbor_values, time_step);

    double A_ij, A_ij_an;
    const auto & system_matrix = pressure_solver.get_system_matrix();

    const double k = data.get_permeability->value(Point<dim>(1,1,1), 1);
    const double phi = data.get_porosity->value(Point<dim>(1,1,1), 1);
    const double mu = data.viscosity_water();
    const double B_w = data.volume_factor_water();
    const double cw = data.compressibility_water();
    const double h = 1;
    // // Compute transmissibility and mass matrix entries
    const double T = 1./mu/B_w*(k/h)*h*h;
    const double B = h*h*h/B_w*phi*cw;

    // // Testing A(0, 0) - two neighbors
    A_ij = system_matrix(0, 0);
    A_ij_an = B/time_step + 2*T;
    AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<DefaultValues::small_number,
                ExcMessage("System matrix is wrong"));
    // Testing A(0, 1) = T
    A_ij = system_matrix(0, 1);
    A_ij_an = -T;
    AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<1e-9,
                ExcMessage("System matrix is wrong"));
    // Testing A(1, 1) - 3 neighbors
    A_ij = system_matrix(1, 1);
    A_ij_an = B/time_step + 3*T;
    AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<1e-9,
                ExcMessage("System matrix is wrong"));
    // Testing A(5, 5) - four neighbors
    A_ij = system_matrix(5, 5);
    A_ij_an = B/time_step + 4*T;
    AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<1e-9,
                ExcMessage("System matrix is wrong"));

    // // -----------------------------------------------------------------------
    pressure_solver.solve();
    // const int n_pressure_iter = pressure_solver.solve();
    // std::cout << "Pressure solver " << n_pressure_iter << " steps" << std::endl;

    // pressure_solver.solution.print(std::cout, 3, true, false);
  } // eom

} // end of namespace

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    dealii::deallog.depth_console (0);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    std::string input_file_name = Parsers::parse_command_line(argc, argv);
    Wings::WingsPressure<3> problem(input_file_name);
    problem.run();
    return 0;
  }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

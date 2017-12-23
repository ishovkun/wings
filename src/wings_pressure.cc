/*
  This test is meant to test the solution for the input given in
  test4x4-homog.prm
  Problem details:
  4x4 cells, with sizes 1x1x1 m^3
  Wells:
  A, vertical, flow rate set
  B, horizontal, diagonally dissects two cells, no values imposed
  C, horizontal, in between two cells, no values imposed

  Testing:
  Wells a b and c are tested for locations and j-indices
  then the entries of the system matrix and rhs vector are tested
  for permeability entries and well rates.
  finally, the system is solved
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
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
    // Triangulation<dim>                triangulation;
    FluidSolvers::PressureSolver<dim>         pressure_solver;
    Data::DataBase<dim>                       data;
    std::string                               input_file;

  };


  template <int dim>
  WingsPressure<dim>::WingsPressure(std::string input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing
                  (Triangulation<dim>::smoothing_on_refinement |
                   Triangulation<dim>::smoothing_on_coarsening)),
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

    for (;cell != endc; cell++)
      // if (!cell->is_artificial)
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

    triangulation.execute_coarsening_and_refinement();
  } // eom


  template <int dim>
  void WingsPressure<dim>::run()
  {
    data.read_input(input_file, /* verbosity = */ 0);
    // data.read_input(input_file, /* verbosity = */ 1);
    // data.print_input();
    // std::cout << "reading mesh file " << data.mesh_file.string() << std::endl;
    // read_mesh();
    // refine_mesh();
    // return;
    // pressure_solver.setup_dofs();
    // // -----------------------------------------------------------------------
    // pressure_solver.solve();
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

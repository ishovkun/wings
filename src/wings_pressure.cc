// #include <deal.II/base/utilities.h>
// #include <deal.II/numerics/data_out.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
// Custom modules
#include <PressureSolver.hpp>

namespace Wings
{
  using namespace dealii;


  template <int dim>
  class WingsPressure
  {
  public:
    WingsPressure();
    // ~WingsPressure();
    void run();

  private:
    void make_mesh();

    Triangulation<dim>                triangulation;
    FluidSolvers::PressureSolver<dim> pressure_solver;
  };


  template <int dim>
  WingsPressure<dim>::WingsPressure()
    :
    pressure_solver(triangulation)
  {}


  // template <int dim>
  // WingsPressure<dim>::~WingsPressure()
  // {}


  template <int dim>
  void WingsPressure<dim>::make_mesh()
  {
    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(1);
  } // eom


  template <int dim>
  void WingsPressure<dim>::run()
  {
    make_mesh();
    pressure_solver.setup_system();

    pressure_solver.solution[0] = 1;
    pressure_solver.solution[1] = 0;
    pressure_solver.solution[2] = 0;
    pressure_solver.solution[3] = 1;
    pressure_solver.assemble_system();
  } // eom

} // end of namespace

// int main(int argc, char *argv[])
int main()
{
  try
  {
    using namespace dealii;

    dealii::deallog.depth_console (0);
    // Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    // std::string input_file_name = parse_command_line(argc, argv);
    Wings::WingsPressure<2> problem;
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

// #include <deal.II/base/utilities.h>
// #include <deal.II/numerics/data_out.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
// Custom modules
#include <PressureSolver.hpp>
#include <Parsers.hpp>

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
    void make_mesh();

    Triangulation<dim>                triangulation;
    FluidSolvers::PressureSolver<dim> pressure_solver;
    Data::DataBase<dim>               data;
    std::string                       input_file;
  };


  template <int dim>
  WingsPressure<dim>::WingsPressure(std::string input_file_name_)
    :
    pressure_solver(triangulation, data),
    input_file(input_file_name_)
  {}


  template <int dim>
  void WingsPressure<dim>::read_mesh()
  {
    GridIn<dim> gridin;
	  gridin.attach_triangulation(triangulation);
    std::cout << "Reading mesh file "
              << data.mesh_file.string()
              << std::endl;
	  std::ifstream f(data.mesh_file.string());

    // typename GridIn<dim>::Format format = GridIn<dim>::ucd;
    // gridin.read(f, format);
	  gridin.read_msh(f);
  }  // eom


  template <int dim>
  void WingsPressure<dim>::make_mesh()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(2);
  } // eom


  template <int dim>
  void WingsPressure<dim>::run()
  {
    data.read_input(input_file);
    read_mesh();
    pressure_solver.setup_system();


    const double k = data.get_permeability->value(Point<dim>(1,1), 1);
    const double phi = data.get_porosity->value(Point<dim>(1,1), 1);
    const double mu = data.viscosity_water();
    const double B_w = data.volume_factor_water();
    const double cw = data.compressibility_water();
    const double h = 1;

    // Compute transmissibility and mass matrix entries
    const double T = 1./mu/B_w*(k/h)*h*h;
    const double B = h*h*h/B_w*phi*cw;
    // test output
    std::cout << "Permeability "
              << k
              << std::endl;
    std::cout << "Porosity "
              << phi
              << std::endl;

    std::cout << "Transmissibility "
              << T
              << std::endl;
    std::cout << "Mass matrix entriy "
              << B
              << std::endl;

    double time = 0;
    double time_step = data.get_time_step(time);


    // pressure_solver.solution[0] = 1;
    // pressure_solver.solution[1] = 0;
    // pressure_solver.solution[2] = 0;
    // pressure_solver.solution[3] = 1;
    // pressure_solver.solution_old = pressure_solver.solution;

    pressure_solver.assemble_system(time_step);
    // pressure_solver.print_system_matrix(1.0/T);
    const auto & system_matrix = pressure_solver.get_system_matrix();
    double A_ij, A_ij_an;

    // Testing A(0, 0) - two neighbors
    A_ij = system_matrix(0, 0);
    A_ij_an = B/time_step + 2*T;
    AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<1e-9,
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
    // Testing A(4, 4) - four neighbors
    A_ij = system_matrix(4, 4);
    A_ij_an = B/time_step + 4*T;
    AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<1e-9,
                ExcMessage("System matrix is wrong"));

  } // eom

} // end of namespace

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    dealii::deallog.depth_console (0);
    // Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    std::string input_file_name = Parsers::parse_command_line(argc, argv);
    Wings::WingsPressure<2> problem(input_file_name);
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

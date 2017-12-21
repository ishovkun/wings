// #include <deal.II/base/utilities.h>
// #include <deal.II/numerics/data_out.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
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
    const auto & pressure_dof_handler = pressure_solver.get_dof_handler();
    const auto & pressure_fe = pressure_solver.get_fe();

    data.locate_wells(pressure_dof_handler, pressure_fe);

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

    data.update_well_productivities();

    const double k = data.get_permeability->value(Point<dim>(1,1,1), 1);
    const double phi = data.get_porosity->value(Point<dim>(1,1,1), 1);
    const double mu = data.viscosity_water();
    const double B_w = data.volume_factor_water();
    const double cw = data.compressibility_water();
    const double h = 1;

    auto & well_A = data.wells[0];
    auto & well_B = data.wells[1];
    auto & well_C = data.wells[2];
    // Well a
    const auto & j_ind_a = well_A.get_productivities();
    std::cout << "Well A J index = " << j_ind_a[0] << std::endl;
    // hand_calculation
    const double well_A_radius = well_A.get_radius();
    const double pieceman_radius =
        0.28*std::sqrt(2*h*h)/(2);
    const double J_index_A =
        2*M_PI*k /
        (std::log(pieceman_radius/well_A_radius));
    std::cout << "J A true " << J_index_A << std::endl;
    AssertThrow(abs(J_index_A - j_ind_a[0])/j_ind_a[0] < DefaultValues::small_number,
                ExcMessage("Wrong J index well A"));
    // Well b
    const auto & j_ind_b = well_B.get_productivities();
    std::cout << "Well B J index = " << j_ind_b[0] << std::endl;
    AssertThrow(abs(j_ind_b[0]) < DefaultValues::small_number*k,
                ExcMessage("This cell J index should be zero!"));
    std::cout << "Well B J index = " << j_ind_b[1] << std::endl;
    std::cout << "Well B J index = " << j_ind_b[2] << std::endl;
    // Well c
    const auto & j_ind_c = well_C.get_productivities();
    std::cout << "Well C J index = " << j_ind_c[0] << std::endl;
    std::cout << "Well C J index = " << j_ind_c[1] << std::endl;

    double time = 0;
    double time_step = data.get_time_step(time);

    data.update_well_controls(time);

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
    std::cout << "Mass matrix entry "
              << B
              << std::endl;

    pressure_solver.solution[0] = 1;
    pressure_solver.solution[1] = 0;
    pressure_solver.solution[2] = 0;
    pressure_solver.solution[3] = 0;
    pressure_solver.solution_old = pressure_solver.solution;

    CellValues::CellValuesBase<dim>
      cell_values(data), neighbor_values(data);
    pressure_solver.assemble_system(cell_values, neighbor_values, time_step);

    // pressure_solver.print_system_matrix(1.0/T);

    double A_ij, A_ij_an;
    const auto & system_matrix = pressure_solver.get_system_matrix();

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

    const auto & rhs_vector = pressure_solver.get_rhs_vector();
    // rhs_vector.print(std::cout, 3, true, false);
    const double rate_A = well_A.get_control().value;
    // std::cout << "Rate A = " << rate_A << std::endl;
    AssertThrow(abs(rhs_vector[4] - rate_A)/rate_A<DefaultValues::small_number,
                    ExcMessage("rhs entry 4 is wrong"));

    const double rhs_0 = B/time_step*pressure_solver.solution[0];
    AssertThrow(abs(rhs_vector[0] - rhs_0)/rhs_0<DefaultValues::small_number,
                ExcMessage("rhs entry 0 is wrong"));

    const int n_pressure_iter = pressure_solver.solve();
    std::cout << "Pressure solver " << n_pressure_iter << " steps" << std::endl;

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

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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/grid/grid_in.h>
// Custom modules
#include <Wellbore.hpp>
#include <Model.hpp>
#include <Reader.hpp>
#include <PressureSolver.hpp>
#include <Parsers.hpp>
#include <CellValues.hpp>

namespace WingTest
{
  using namespace dealii;


  template <int dim>
  class TestSPO
  {
  public:
    TestSPO(std::string);
    // ~TestSPO();
    void read_mesh();
    void run();

  private:
    MPI_Comm                                  mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    ConditionalOStream                        pcout;
    Model::Model<dim>                         data;
    FluidSolvers::PressureSolver<dim>         pressure_solver;
    std::string                               input_file;
  };


  template <int dim>
  TestSPO<dim>::TestSPO(std::string input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    data(mpi_communicator, pcout),
    pressure_solver(mpi_communicator, triangulation, data, pcout),
    input_file(input_file_name_)
  {}


  template <int dim>
  void TestSPO<dim>::read_mesh()
  {
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(data.mesh_file.string());

    // typename GridIn<dim>::Format format = GridIn<dim>::ucd;
    // gridin.read(f, format);
    gridin.read_msh(f);
  }  // eom


  template <int dim>
  void TestSPO<dim>::run()
  {
    Parsers::Reader reader(pcout, data);
    reader.read_input(input_file, /* verbosity= */0);
    // data.print_input();
    read_mesh();
    pressure_solver.setup_dofs();

    auto & well_A = data.wells[0];
    auto & well_B = data.wells[1];
    auto & well_C = data.wells[2];

    // true values that should be given by solution
    // data
    const double k = data.get_permeability->value(Point<dim>(1,1,1), 1);
    const double phi = data.get_porosity->value(Point<dim>(1,1,1), 1);
    const double mu = data.viscosity_water();
    const double B_w = data.volume_factor_water();
    const double cw = data.compressibility_water();
    const double h = 1;
    // // Compute transmissibility and mass matrix entries
    const double T = 1./mu/B_w*(k/h)*h*h;
    const double B = h*h*h/B_w*phi*cw;
    // Well cell centers
    const Point<3> cell_A_t = Point<3>(1.5, 0.5, 0.0);
    std::vector<Point<3>> cells_B_t(3);
    cells_B_t[0] = Point<3>(1.5, 1.5, 0);
    cells_B_t[1] = Point<3>(1.5, 2.5, 0);
    cells_B_t[2] = Point<3>(2.5, 1.5, 0);
    // I don't like this entry since the well is parallel to the face
    // we'll see what the J index gives
    std::vector<Point<3>> cells_C_t(3);
    cells_C_t[0] = Point<3>(1.5, 0.5, 0);
    cells_C_t[1] = Point<3>(2.5, 0.5, 0);
    cells_C_t[2] = Point<3>(3.5, 0.5, 0);
    // well A productivity
    const double well_A_length = h;
    const double well_B_length = h*std::sqrt(2.);
    const double well_C_length = 2*h;
    const double well_A_radius = well_A.get_radius();
    const double well_B_radius = well_B.get_radius();
    const double well_C_radius = well_C.get_radius();
    const double pieceman_radius = 0.28*std::sqrt(2*h*h)/2;
    const double J_index_A =
      2*M_PI*k*well_A_length / (std::log(pieceman_radius/well_A_radius));
    const double J_index_B =
      2*M_PI*k*well_B_length/ (std::log(pieceman_radius/well_B_radius));
    const double J_index_C =
        2*M_PI*k*well_C_length/ (std::log(pieceman_radius/well_C_radius));

    // std::cout << "rB " << well_B_radius << std::endl;
    // std::cout << "rC " << well_C_radius << std::endl;

    const double delta = DefaultValues::small_number;

    //  What code gives
    const auto & pressure_dof_handler = pressure_solver.get_dof_handler();
    const auto & pressure_fe = pressure_solver.get_fe();
    data.locate_wells(pressure_dof_handler, pressure_fe);
    data.update_well_productivities();

    // some init values
    pressure_solver.solution = 0;
    pressure_solver.solution[0] = 1;
    pressure_solver.old_solution = pressure_solver.solution;

    double time = 0;
    double time_step = data.min_time_step;

    data.update_well_controls(time);

    CellValues::CellValuesBase<dim>
      cell_values(data), neighbor_values(data);
    pressure_solver.assemble_system(cell_values, neighbor_values, time_step);

    // for (auto & id : data.get_well_ids())
    // {
    //   std::cout << "well_id " << id << std::endl;
    //   auto & well = data.wells[id];

    //   std::cout << "Real locations"  << std::endl;
    //   for (auto & loc : well.get_locations())
    //     std::cout << loc << std::endl;

    //   std::cout << "Assigned locations"  << std::endl;
    //   for (auto & cell : well.get_cells())
    //     std::cout << cell->center() << std::endl;

    //   std::cout << std::endl;
    // }

    // A test for properly placing wells into cells
    const auto & cells_A = well_A.get_cells();
    AssertThrow(cells_A.size() == 1,
                ExcDimensionMismatch(cells_A.size(), 1));
    AssertThrow(cells_A[0]->center().distance(cell_A_t) < delta,
                ExcMessage("well A located wrong"));
    const auto & cells_B = well_B.get_cells();
    AssertThrow(cells_B.size() == 3,
                ExcDimensionMismatch(cells_B.size(), 3));
    for (unsigned int i=0; i< cells_B.size(); i++)
    {
      AssertThrow(cells_B[i]->center().distance(cells_B_t[i]) < delta,
                  ExcMessage("well B located wrong"));
    }
    const auto & cells_C = well_C.get_cells();
    AssertThrow(cells_C.size() == 3,
                ExcDimensionMismatch(cells_C.size(), 2));
    for (unsigned int i=0; i< cells_C.size(); i++)
    {
      AssertThrow(cells_C[i]->center().distance(cells_C_t[i]) < delta,
                  ExcMessage("well C located wrong"));
    }

    // Cell J indices
    // Well a
    const auto & j_ind_a = well_A.get_productivities();
    // std::cout << "Well A J index = " << j_ind_a[0] << std::endl;
    // std::cout << "J A true " << J_index_A << std::endl;
    AssertThrow(abs(J_index_A - j_ind_a[0])/j_ind_a[0] < delta,
                ExcMessage("Wrong J index well A"));
    // Well b
    const auto & j_ind_b = well_B.get_productivities();
    const double j_b_total = j_ind_b[0] + j_ind_b[1] + j_ind_b[2];
    AssertThrow(abs(J_index_B - j_b_total)/J_index_B <
                DefaultValues::small_number_geometry*10,
                ExcMessage("Wrong J index well B"));
    // Well c
    const auto & j_ind_c = well_C.get_productivities();
    const double j_c_total = j_ind_c[0] + j_ind_c[1] + j_ind_c[2];
    AssertThrow(abs(J_index_C - j_c_total)/J_index_C <
                DefaultValues::small_number_geometry*10,
                ExcMessage("Wrong J index well C"));

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
    // std::cout << "rhs_vector[4] = " << rhs_vector[4] << std::endl;
    AssertThrow(abs(rhs_vector[4] - rate_A)/rate_A<DefaultValues::small_number,
                    ExcMessage("rhs entry 4 is wrong"));

    const double rhs_0 = B/time_step*pressure_solver.solution[0];
    AssertThrow(abs(rhs_vector[0] - rhs_0)/rhs_0<DefaultValues::small_number,
                ExcMessage("rhs entry 0 is wrong"));

    pressure_solver.solve();
    // std::cout << "Pressure solver " << n_pressure_iter << " steps" << std::endl;

    // pressure_solver.solution.print(std::cout, 3, true, false);
  } // eom

} // end of namespace

// int main(int argc, char *argv[])
int main(int argc, char *argv[])
{
    using namespace dealii;
    dealii::deallog.depth_console (0);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    std::string input_file_name = SOURCE_DIR "/../data/sf-4x4.data";
    WingTest::TestSPO<3> problem(input_file_name);
    problem.run();
    return 0;
}

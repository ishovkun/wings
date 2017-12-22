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
    void refine_mesh();

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
    data.read_input(input_file, /* verbosity = */ 1);
    // data.print_input();
    read_mesh();
    refine_mesh();
    // return;
    pressure_solver.setup_dofs();

    auto & well_A = data.wells[0];
    auto & well_B = data.wells[1];
    auto & well_C = data.wells[2];

    // // true values that should be given by solution
    // // data
    const double k = data.get_permeability->value(Point<dim>(1,1,1), 1);
    // const double phi = data.get_porosity->value(Point<dim>(1,1,1), 1);
    // const double mu = data.viscosity_water();
    // const double B_w = data.volume_factor_water();
    // const double cw = data.compressibility_water();
    const double h = 1;
    // // // Compute transmissibility and mass matrix entries
    // const double T = 1./mu/B_w*(k/h)*h*h;
    // const double B = h*h*h/B_w*phi*cw;
    // // Well cell centers
    // const Point<3> cell_A_t = Point<3>(1.5, 0.5, 0.0);
    std::vector<Point<3>> cells_B_t(5);
    cells_B_t[0] = Point<3>(1.5, 1.5, 0);
    cells_B_t[1] = Point<3>(2.5, 1.5, 0);
    cells_B_t[2] = Point<3>(1.25, 2.25, 0.25);
    cells_B_t[3] = Point<3>(1.75, 2.25, 0.25);
    cells_B_t[4] = Point<3>(1.25, 2.25, -0.25);



    // // I don't like this entry since the well is parallel to the face
    // // we'll see what the J index gives
    // std::vector<Point<3>> cells_C_t(3);
    // cells_C_t[0] = Point<3>(1.5, 0.5, 0);
    // cells_C_t[1] = Point<3>(2.5, 0.5, 0);
    // cells_C_t[2] = Point<3>(3.5, 0.5, 0);
    // // well A productivity
    // const double well_A_length = h;
    const double well_B_length = h*std::sqrt(2.);
    // const double well_C_length = 2*h;
    // const double well_A_radius = well_A.get_radius();
    const double well_B_radius = well_B.get_radius();
    // const double well_C_radius = well_C.get_radius();
    // const double J_index_A =
    //   2*M_PI*k*well_A_length / (std::log(pieceman_radius/well_A_radius));

    /* this is where it gets funky!!!!
       Since well B is partially in a large cell (h=1) and partially in a
       refined cell, it's well index is now different!
     */
    const double pieceman_radius_coarse = 0.28*std::sqrt(2*h*h)/2;
    const double pieceman_radius_fine = 0.28*std::sqrt(2*h/2*h/2)/2;
    const double J_index_B =
        2*M_PI*k*well_B_length/2/(std::log(pieceman_radius_coarse/well_B_radius)) +
        2*M_PI*k*well_B_length/2/(std::log(pieceman_radius_fine/well_B_radius));

    // const double J_index_C =
    //     2*M_PI*k*well_C_length/ (std::log(pieceman_radius/well_C_radius));

    // // std::cout << "rB " << well_B_radius << std::endl;
    // // std::cout << "rC " << well_C_radius << std::endl;

    const double delta = DefaultValues::small_number;

    // //  What code gives
    const auto & pressure_dof_handler = pressure_solver.get_dof_handler();
    const auto & pressure_fe = pressure_solver.get_fe();
    data.locate_wells(pressure_dof_handler, pressure_fe);
    data.update_well_productivities();

    // some init values
    pressure_solver.solution = 0;
    pressure_solver.solution[0] = 1;
    pressure_solver.solution_old = pressure_solver.solution;

    double time = 1;
    double time_step = data.get_time_step(time);

    data.update_well_controls(time);

    const auto & well_B_control = well_B.get_control();
    AssertThrow(well_B_control.type == Schedule::WellControlType::pressure_control,
                ExcMessage("Wrong control of well B"));
    // CellValues::CellValuesBase<dim>
    //   cell_values(data), neighbor_values(data);
    // pressure_solver.assemble_system(cell_values, neighbor_values, time_step);

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

    // // A test for properly placing wells into cells
    const auto & cells_B = well_B.get_cells();
    for (unsigned int i=0; i< cells_B_t.size(); i++)
    {
      AssertThrow(cells_B[i]->center().distance(cells_B_t[i]) < delta,
                  ExcMessage("well B located wrong"));
    }

    // Cell J indices
    // Well b
    const auto & j_ind_b = well_B.get_productivities();
    const double j_b_total = Math::sum(j_ind_b);
    // std::cout << "J_b true = " << J_index_B << "\t"
    //           << "J_b = " << j_b_total
    //           << std::endl;

    // for (unsigned int i=0; i<j_ind_b.size(); i++)
    //   std::cout << "J_b part = " << j_ind_b[i] << "\t";
    // std::cout << std::endl;

    AssertThrow(abs(J_index_B - j_b_total)/J_index_B <
                DefaultValues::small_number_geometry*10,
                ExcMessage("Wrong J index well B"));

    // Wells A and C are the same as in the last test

    // double A_ij, A_ij_an;
    // const auto & system_matrix = pressure_solver.get_system_matrix();
    // system_matrix.print_formatted(std::cout);

    // // // Testing A(0, 0) - two neighbors
    // A_ij = system_matrix(0, 0);
    // A_ij_an = B/time_step + 2*T;
    // AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<DefaultValues::small_number,
    //             ExcMessage("System matrix is wrong"));
    // // Testing A(0, 1) = T
    // A_ij = system_matrix(0, 1);
    // A_ij_an = -T;
    // AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<1e-9,
    //             ExcMessage("System matrix is wrong"));
    // // Testing A(1, 1) - 3 neighbors
    // A_ij = system_matrix(1, 1);
    // A_ij_an = B/time_step + 3*T;
    // AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<1e-9,
    //             ExcMessage("System matrix is wrong"));
    // // Testing A(5, 5) - four neighbors
    // A_ij = system_matrix(5, 5);
    // A_ij_an = B/time_step + 4*T;
    // AssertThrow(abs(A_ij - A_ij_an)/abs(A_ij_an)<1e-9,
    //             ExcMessage("System matrix is wrong"));

    // const auto & rhs_vector = pressure_solver.get_rhs_vector();
    // // rhs_vector.print(std::cout, 3, true, false);
    // const double rate_A = well_A.get_control().value;
    // // std::cout << "Rate A = " << rate_A << std::endl;
    // AssertThrow(abs(rhs_vector[4] - rate_A)/rate_A<DefaultValues::small_number,
    //                 ExcMessage("rhs entry 4 is wrong"));

    // const double rhs_0 = B/time_step*pressure_solver.solution[0];
    // AssertThrow(abs(rhs_vector[0] - rhs_0)/rhs_0<DefaultValues::small_number,
    //             ExcMessage("rhs entry 0 is wrong"));

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

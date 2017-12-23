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
    const double phi = data.get_porosity->value(Point<dim>(1,1,1), 1);
    const double mu = data.viscosity_water();
    const double B_w = data.volume_factor_water();
    const double cw = data.compressibility_water();
    const double h = 1;
    // Compute transmissibility and mass matrix entries
    const double T_coarse_coarse = 1./mu/B_w*(k/h)*h*h;
    const double T_fine_fine = 1./mu/B_w*(2*k/h)*h*h/4;
    const double T_fine_coarse = 1./mu/B_w*(k/(h/2 +h/4))*h*h/4;
    const double B = h*h*h/B_w*phi*cw;
    const double B_fine = h*h*h/8/B_w*phi*cw;
    // // Well cell centers
    // const Point<3> cell_A_t = Point<3>(1.5, 0.5, 0.0);
    std::vector<Point<3>> cells_B_t(5);
    cells_B_t[0] = Point<3>(1.5, 1.5, 0);
    cells_B_t[1] = Point<3>(2.5, 1.5, 0);
    cells_B_t[2] = Point<3>(1.25, 2.25, 0.25);
    cells_B_t[3] = Point<3>(1.75, 2.25, 0.25);
    cells_B_t[4] = Point<3>(1.25, 2.25, -0.25);

    // // well A productivity
    const double well_B_length = h*std::sqrt(2.);
    const double well_B_radius = well_B.get_radius();

    /* this is where it gets funky!!!!
       Since well B is partially in a large cell (h=1) and partially in a
       refined cell, it's well index is now different!
     */
    const double pieceman_radius_coarse = 0.28*std::sqrt(2*h*h)/2;
    const double pieceman_radius_fine = 0.28*std::sqrt(2*h/2*h/2)/2;
    const double J_index_B_coarse =
        2*M_PI*k*well_B_length/2/(std::log(pieceman_radius_coarse/well_B_radius));
    const double J_index_B_fine =
        2*M_PI*k*well_B_length/2/(std::log(pieceman_radius_fine/well_B_radius));
    const double J_index_B = J_index_B_coarse + J_index_B_fine;

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

    const auto & well_A_control = well_A.get_control();
    AssertThrow(well_A_control.value == 0.0,
                ExcMessage("Wrong control of well A"));
    AssertThrow(well_A_control.type == Schedule::WellControlType::flow_control_total,
                ExcMessage("Wrong control of well A"));
    const auto & well_B_control = well_B.get_control();
    AssertThrow(well_B_control.type == Schedule::WellControlType::pressure_control,
                ExcMessage("Wrong control of well B"));

    CellValues::CellValuesBase<dim>
      cell_values(data), neighbor_values(data);
    pressure_solver.assemble_system(cell_values, neighbor_values, time_step);

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

    // -----------------------------------------------------------------------
    // System matrix
    const auto & system_matrix = pressure_solver.get_system_matrix();
    // system_matrix.print_formatted(std::cout,
    //                               /* precision = */ 3,
    //                               /*scientific = */ false,
    //                               /*width = */ 0,
    //                               /*zero_string = */ " ",
    //                               // /*denominator = */ time_step/B);
    //                               /*denominator = */ 1./T_coarse_coarse);
    std::cout << "T_cc = " << T_coarse_coarse << std::endl;
    std::cout << "T_ff/T_cc = " << T_fine_fine/T_coarse_coarse << std::endl;
    std::cout << "T_fc/T_cc = " << T_fine_coarse/T_coarse_coarse << std::endl;

    const double eps = DefaultValues::small_number;
    const double eps_geo = DefaultValues::small_number_geometry;
    // Test coarse_coarse transmissibilities
    std::vector<int> is = {0, 1, 2, 4, 7, 8, 9, 11, 12, 13};
    std::vector<double> js;
    for (auto & i : is)
    {
      double A_ij = system_matrix(i, i+1);
      AssertThrow(abs(A_ij + T_coarse_coarse)/T_coarse_coarse < eps, ExcMessage("wrong"));
    }
    is = {1, 2, 3, 5, 8, 9, 10, 12, 13, 14};
    for (auto & i : is)
    {
      double A_ij = system_matrix(i, i-1);
      AssertThrow(abs(A_ij + T_coarse_coarse)/T_coarse_coarse < eps, ExcMessage("wrong"));
    }
    is = {0, 1, 7, 8, 9, 10};
    for (auto & i : is)
    {
      double A_ij = system_matrix(i, i+4);
      AssertThrow(abs(A_ij + T_coarse_coarse)/T_coarse_coarse < eps, ExcMessage("wrong"));
    }

    // Test fine fine entries
    is = {15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21};
    js = {16, 17, 19, 15, 18, 20, 15, 18, 21, 16, 17, 22, 15, 20, 21, 16, 19, 22, 17, 19, 22};
    for (unsigned int pp=0; pp < is.size(); pp++)
    {
      double A_ij = system_matrix(is[pp], js[pp]);
      AssertThrow(abs(A_ij + T_fine_fine)/T_fine_fine < eps, ExcMessage("wrong"));
    }

    // Test fine coarse entries
    is = {2,  2,  2,  2,  5,  5,  5,  5,  6,  6,  6,  6,  9,  9,  9,  9,  15, 15, 16, 16};
    js = {15, 17, 19, 21, 15, 16, 17, 18, 19, 20, 21, 22, 16, 18, 20, 22, 2,  5,  5,  9};
    for (unsigned int pp=0; pp < is.size(); pp++)
    {
      double A_ij = system_matrix(is[pp], js[pp]);
      AssertThrow(abs(A_ij + T_fine_coarse)/T_fine_coarse < eps, ExcMessage("wrong"));
    }

    // system_matrix.print(std::cout, false, /*diagonal_first*/ true);
    // system_matrix.print(std::cout);
    // Test diagonal
    double A_ii_an, A_ii;
    // (0, 0)
    A_ii_an = B/time_step + 2*T_coarse_coarse;
    A_ii = system_matrix(0, 0);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 0_0"));
    // (1, 1)
    A_ii_an = B/time_step + 3*T_coarse_coarse;
    A_ii = system_matrix(1, 1);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 1_1"));
    // (2, 2)
    A_ii_an = B/time_step + 2*T_coarse_coarse + 4*T_fine_coarse;
    A_ii = system_matrix(2, 2);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 2_2"));
    // (3, 3)
    A_ii_an = B/time_step + 2*T_coarse_coarse;
    A_ii = system_matrix(3, 3);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 3_3"));
    // (4, 4)
    A_ii_an = B/time_step + 3*T_coarse_coarse;
    A_ii = system_matrix(4, 4);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 4_4"));
    // (5, 5) Due to rounoff in well location there is addition from well B
    // Tolerance is geometric!
    A_ii_an = B/time_step + 3*T_coarse_coarse + 4*T_fine_coarse;
    A_ii = system_matrix(5, 5);
    // std::cout << "a55 " << A_ii << std::endl;
    // std::cout << "a55_an " << A_ii_an << std::endl;
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps_geo, ExcMessage("wrong 5_5"));
    // (6, 6)
    A_ii_an = B/time_step + 2*T_coarse_coarse + 4*T_fine_coarse;
    A_ii = system_matrix(6, 6);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 6_6"));
    // (7, 7)
    A_ii_an = B/time_step + 3*T_coarse_coarse;
    A_ii = system_matrix(7, 7);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 7_7"));
    // (8, 8)
    A_ii_an = B/time_step + 4*T_coarse_coarse + J_index_B_coarse;
    A_ii = system_matrix(8, 8);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 8_8"));
    // (9, 9)
    A_ii_an = B/time_step + 3*T_coarse_coarse + 4*T_fine_coarse;
    A_ii = system_matrix(9, 9);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 9_9"));
    // (10, 10)
    A_ii_an = B/time_step + 3*T_coarse_coarse;
    A_ii = system_matrix(10, 10);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 10_10"));
    // (11, 11)
    A_ii_an = B/time_step + 2*T_coarse_coarse;
    A_ii = system_matrix(11, 11);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 11_11"));
    // (12, 12)
    A_ii_an = B/time_step + 3*T_coarse_coarse;
    A_ii = system_matrix(12, 12);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 12_12"));
    // (13, 13)
    A_ii_an = B/time_step + 3*T_coarse_coarse;
    A_ii = system_matrix(13, 13);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 13_13"));
    // (14, 14)
    A_ii_an = B/time_step + 2*T_coarse_coarse;
    A_ii = system_matrix(14, 14);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 14_14"));
    // (15, 15) left_front refined cell at top
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine;
    A_ii = system_matrix(15, 15);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 15_15"));
    // (16, 16) right-front refined cell at top
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine + J_index_B_fine;
    A_ii = system_matrix(16, 16);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 16_16"));
    // (17, 17) left_front refined cell at bottom
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine;
    A_ii = system_matrix(17, 17);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 17_17"));
    // (18, 18) left_front refined cell at bottom
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine;
    A_ii = system_matrix(18, 18);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 18_18"));
    // (19, 19)
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine;
    A_ii = system_matrix(19, 19);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 19_19"));
    // (20, 20)
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine;
    A_ii = system_matrix(20, 20);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 20_20"));
    // (21, 21)
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine;
    A_ii = system_matrix(21, 21);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 21_21"));
    // (22, 22)
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine;
    A_ii = system_matrix(22, 22);
    AssertThrow(abs(A_ii - A_ii_an)/A_ii_an < eps, ExcMessage("wrong 22_22"));

    // -----------------------------------------------------------------------
    // RHS vector
    const auto & rhs_vector = pressure_solver.get_rhs_vector();
    // rhs_vector.print(std::cout, 3, true, false);
    double rhs_an;

    // indices that should be zero
    // indices 5 should be zero like geometric small value.
    // we compare it with index in cell 8
    is = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14};
    for (auto &i : is)
      AssertThrow(abs(rhs_vector[i]) < eps, ExcMessage("wrong rhs "));

    // const double pressure_B = well_B.get_control().value;
    // this guy exhists only in fine cells
    const double g_vector_entry = data.density_sc_water()/B_w/B_w*data.gravity() *
        T_fine_fine * (h/2);
    // 15
    rhs_an = -g_vector_entry;
    std::cout<< "rhs an = "<< rhs_an << std::endl;
    std::cout<< "rhs num = "<< rhs_vector[15] << std::endl;
    AssertThrow(abs(rhs_vector[15] - rhs_an)/abs(rhs_an) < eps,
                    ExcMessage("rhs entry 15 is wrong"));

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

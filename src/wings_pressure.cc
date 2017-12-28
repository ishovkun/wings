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
    Data::DataBase<dim>                       data;
    FluidSolvers::PressureSolver<dim>         pressure_solver;
    std::string                               input_file;
    TimerOutput                               computing_timer;
  };


  template <int dim>
  WingsPressure<dim>::WingsPressure(std::string input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    data(mpi_communicator, pcout),
    pressure_solver(mpi_communicator, triangulation, data, pcout),
    input_file(input_file_name_),
    computing_timer(mpi_communicator, pcout,
                    TimerOutput::summary, TimerOutput::wall_times)
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

    auto & well_A = data.wells[0];
    auto & well_B = data.wells[1];
    // auto & well_C = data.wells[2];

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

    // // true values that should be given by solution
    // // data
    const double k = data.get_permeability->value(Point<dim>(1,1,1), 1);
    const double phi = data.get_porosity->value(Point<dim>(1,1,1), 1);
    const double mu = data.viscosity_water();
    const double B_w = data.volume_factor_water();
    const double cw = data.compressibility_water();
    const double h = 1;
    // Compute transmissibility and mass matrix entries
    // const double T_coarse_coarse = 1./mu/B_w*(k/h)*h*h;
    // const double T_fine_fine = 1./mu/B_w*(2*k/h)*h*h/4;
    // const double T_fine_coarse = 1./mu/B_w*(k/(h/2 +h/4))*h*h/4;
    // const double B = h*h*h/B_w*phi*cw;
    // const double B_fine = h*h*h/8/B_w*phi*cw;
    const double pieceman_radius_coarse = 0.28*std::sqrt(2*h*h)/2;
    const double pieceman_radius_fine = 0.28*std::sqrt(2*h/2*h/2)/2;
    const double well_B_length = h*std::sqrt(2.);
    const double well_B_radius = well_B.get_radius();
    const double J_index_B_coarse =
      2*M_PI*k*well_B_length/2/(std::log(pieceman_radius_coarse/well_B_radius));
    const double J_index_B_fine =
      2*M_PI*k*well_B_length/2/(std::log(pieceman_radius_fine/well_B_radius));
    const double J_index_B = J_index_B_coarse + J_index_B_fine;

    const auto & j_ind_b = well_B.get_productivities();
    double j_b_total = 0;
    if (j_ind_b.size() > 0)
      j_b_total = Math::sum(j_ind_b);
    j_b_total = Utilities::MPI::sum(j_b_total, mpi_communicator);

    // std::cout << "J_b true = " << J_index_B << "\t"
    //           << "J_b = " << j_b_total
    //           << std::endl;
    AssertThrow(abs(J_index_B - j_b_total)/J_index_B <
                DefaultValues::small_number_geometry*10,
                ExcMessage("Wrong J index well B"));

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

    const auto & rhs_vector = pressure_solver.get_rhs_vector();
    const double rate_A = well_A.get_control().value;
    std::cout << "Rate A = " << rate_A << std::endl;
    std::cout << "rhs_vector[4] = " << rhs_vector[4] << std::endl;
    AssertThrow(abs(rhs_vector[4] - rate_A)/rate_A<DefaultValues::small_number,
                ExcMessage("rhs entry 4 is wrong"));

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

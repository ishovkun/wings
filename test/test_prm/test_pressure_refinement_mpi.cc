/*
  This test is similar to test_pressure_refinement
  but it runs on 2 MPI processes
  Parameters given in test4x4-homog.prm
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/distributed/tria.h>

// Custom modules
#include <Model.hpp>
#include <Reader.hpp>
#include <Wellbore.hpp>
#include <SolverIMPES.hpp>
#include <Parsers.hpp>
#include <CellValues/CellValuesBase.hpp>
#include <FEFunction/FEFunction.hpp>

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
    Model::Model<dim>                         model;
    std::string                               input_file;
    // TimerOutput                               computing_timer;
  };


  template <int dim>
  WingsPressure<dim>::WingsPressure(std::string input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    model(mpi_communicator, pcout),
    input_file(input_file_name_)
    // computing_timer(mpi_communicator, pcout,
    //                 TimerOutput::summary, TimerOutput::wall_times)
  {}


  template <int dim>
  void WingsPressure<dim>::read_mesh()
  {
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(model.mesh_config.file.string());

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
    Parsers::Reader reader(pcout, model);
    reader.read_input(input_file, /* verbosity= */0);
    read_mesh();

    refine_mesh();
    // return;
    FluidSolvers::SolverIMPES<dim> fluid_solver(mpi_communicator,
                                                triangulation,
                                                model, pcout);

    fluid_solver.setup_dofs();

    // auto & well_A = model.wells[0];
    auto & well_B = model.wells[1];
    // auto & well_C = model.wells[2];

    // const auto & pressure_dof_handler = fluid_solver.get_dof_handler();
    // const auto & pressure_fe = fluid_solver.get_fe();
    model.locate_wells(fluid_solver.get_dof_handler());
    // model.update_well_productivities();

    // for (auto & id : model.get_well_ids())
    // {
    //   std::cout << "well_id " << id << std::endl;
    //   auto & well = model.wells[id];

    //   std::cout << "Real locations"  << std::endl;
    //   for (auto & loc : well.get_locations())
    //     std::cout << loc << std::endl;

    //   std::cout << "Assigned locations"  << std::endl;
    //   for (auto & cell : well.get_cells())
    //     std::cout << cell->center() << std::endl;

    //   std::cout << std::endl;
    // }

    // // true values that should be given by solution
    // // model
    const double k = model.get_permeability->value(Point<dim>(1,1,1), 1);
    const double phi = model.get_porosity->value(Point<dim>(1,1,1), 1);
    // const double mu = model.viscosity_water();
    // const double B_w = model.volume_factor_water();
    // const double cw = model.compressibility_water();
    const double mu = 1e-3;
    const double B_w = 1;
    const double cw = 5e-10;
    const double h = 1;
    // Compute transmissibility and mass matrix entries
    const double T_coarse_coarse = 1./mu/B_w*(k/h)*h*h;
    const double T_fine_fine = 1./mu/B_w*(2*k/h)*h*h/4;
    const double T_fine_coarse = 1./mu/B_w*(k/(h/2 +h/4))*h*h/4;
    const double B = h*h*h/B_w*phi*cw;
    const double B_fine = h*h*h/8/B_w*phi*cw;
    const double pieceman_radius_coarse = 0.28*std::sqrt(2*h*h)/2;
    const double pieceman_radius_fine = 0.28*std::sqrt(2*h/2*h/2)/2;
    const double well_B_length = h*std::sqrt(2.);
    const double well_B_radius = well_B.get_radius();
    const double J_index_B_coarse =
      1./mu*2*M_PI*k*well_B_length/2/(std::log(pieceman_radius_coarse/well_B_radius));
    const double J_index_B_fine =
      1./mu*2*M_PI*k*well_B_length/2/(std::log(pieceman_radius_fine/well_B_radius));
    const double J_index_B = J_index_B_coarse + J_index_B_fine;


    double time = 1;
    double time_step = model.min_time_step;

    // some init values
    // pressure
    fluid_solver.solution = 0;
    fluid_solver.solution[0] = 1;
    fluid_solver.pressure_relevant = fluid_solver.solution;
    // saturation
    fluid_solver.solution[0] = 1;
    fluid_solver.saturation_relevant[0] = fluid_solver.solution;


    model.update_well_controls(time);

    FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
        pressure_function(fluid_solver.get_dof_handler(),
                          fluid_solver.pressure_relevant);
    FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
        saturation_function(fluid_solver.get_dof_handler(),
                            fluid_solver.saturation_relevant);

    model.update_well_productivities(pressure_function, saturation_function);

    const auto & j_ind_b = well_B.get_productivities();

    // double j_b_total = 0;
    //   j_b_total = Math::sum(j_ind_b);
    double j_b_total = 0;
    if (j_ind_b.size() > 0)
    {
      for (const auto & jj : j_ind_b)
        j_b_total += jj[0];
    }

    j_b_total = Utilities::MPI::sum(j_b_total, mpi_communicator);

    // std::cout << "J_b true = " << J_index_B << "\t"
    //           << "J_b = " << j_b_total
    //           << std::endl;
    AssertThrow(abs(J_index_B - j_b_total)/J_index_B <
                DefaultValues::small_number_geometry*10,
                ExcMessage("Wrong J index well B"));
    // pcout << "wells checked " << std::endl;


    CellValues::CellValuesBase<dim>
      cell_values(model), neighbor_values(model);
    fluid_solver.assemble_pressure_system(cell_values, neighbor_values,
                                          time_step);

    // -----------------------------------------------------------------------
    // RHS vector
    const auto & rhs_vector = fluid_solver.get_rhs_vector();
    double rhs_an;

    // const auto & dfh = fluid_solver.get_dof_handler();
    // typename DoFHandler<dim>::active_cell_iterator
    //     cell = dfh.begin_active();
    // int index = 0;
    // for (cell; cell<dfh.end(); ++cell)
    // {
    //   if (cell->is_locally_owned())
    //     std::cout << index << "\t" << cell->center() << std::endl;
    //   index++;
    // }

    // indices that should be zero
    // indices 5 should be zero like geometric small value.
    // we compare it with index in cell 8
    const auto & lo_dofs = fluid_solver.locally_owned_dofs;
    const double eps = DefaultValues::small_number;
    std::vector<int> is = {1, 2, 3, 4, 14, 15, 17, 18, 19, 20, 21, 22};
    for (const auto &i : is)
      if (std::find(lo_dofs.begin(), lo_dofs.end(), i) != lo_dofs.end())
      {
        // std::cout << "dof = " << i << "\t"
        //     << "rhs_num " << rhs_vector[i] << "\t"
        //           << "rhs_an " << rhs_an << std::endl;
        AssertThrow(abs(rhs_vector[i]) < DefaultValues::small_number_geometry,
                    ExcMessage("wrong rhs " + std::to_string(i)));
      }

    // index 5 has a small portion of the well and needs tesning!!!

    if (std::find(lo_dofs.begin(), lo_dofs.end(), 0) != lo_dofs.end())
      AssertThrow(abs(rhs_vector[0]-B/time_step) < DefaultValues::small_number_geometry,
                  ExcMessage("wrong rhs " + std::to_string(0)));

    // fine cell with  well B
    const double pressure_B = well_B.get_control().value;
    // this guy exhists only in fine cells
    const double g_vector_entry = model.density_sc_water()/B_w/B_w*model.gravity() *
      T_fine_fine * (h/2);

    int dof = 16;
    rhs_an = pressure_B * J_index_B_coarse;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
      AssertThrow(abs(rhs_vector[dof] - rhs_an)/abs(rhs_an) < eps,
                  ExcMessage("wrong rhs entry"+ std::to_string(dof)));

    dof = 7;
    rhs_an = -g_vector_entry + J_index_B_fine*pressure_B;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
    {
      // std::cout << "rhs_num " << rhs_vector[dof] << "\t"
      //           << "rhs_an " << rhs_an << std::endl;
      AssertThrow(abs(rhs_vector[dof] - rhs_an)/abs(rhs_an) < DefaultValues::small_number_geometry,
                  ExcMessage("wrong rhs entry"+ std::to_string(dof)));
    }

    dof = 9;
    rhs_an = +g_vector_entry;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
      AssertThrow(abs(rhs_vector[dof] - rhs_an)/abs(rhs_an) < eps,
                  ExcMessage("wrong rhs entry"+ std::to_string(dof)));
    dof = 12;
    rhs_an = +g_vector_entry;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
      AssertThrow(abs(rhs_vector[dof] - rhs_an)/abs(rhs_an) < eps,
                  ExcMessage("wrong rhs entry"+ std::to_string(dof)));

    // -----------------------------------------------------------------------
    // System matrix
    const auto & system_matrix = fluid_solver.get_system_matrix();

    double A_ii_an;
    // (0, 0)
    A_ii_an = B/time_step + 2*T_coarse_coarse;
    dof = 0;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
      AssertThrow(abs(system_matrix(dof, dof) - A_ii_an)/A_ii_an < eps,
                  ExcMessage("wrong "+std::to_string(dof)+", "+std::to_string(dof)));
    // (1, 1)
    A_ii_an = B/time_step + 3*T_coarse_coarse;
    dof = 1;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
      AssertThrow(abs(system_matrix(dof, dof) - A_ii_an)/A_ii_an < eps,
                  ExcMessage("wrong "+std::to_string(dof)+", "+std::to_string(dof)));
    // (2, 2)
    A_ii_an = B/time_step + 2*T_coarse_coarse + 4*T_fine_coarse;
    dof = 2;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
    {
      // std::cout << "a_22 = " << system_matrix(2, 2) << "\t"
      //           << "a_22_true = " << A_ii_an
      //           << std::endl;
      AssertThrow(abs(system_matrix(dof, dof) - A_ii_an)/A_ii_an < eps,
                  ExcMessage("wrong "+std::to_string(dof)+", "+std::to_string(dof)));
    }
    // (10 10)
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine;
    dof = 10;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
      AssertThrow(abs(system_matrix(dof, dof) - A_ii_an)/A_ii_an < eps,
                  ExcMessage("wrong "+std::to_string(dof)+", "+std::to_string(dof)));
    // (7, 7) fine cell with wellbore B
    A_ii_an = B_fine/time_step + 2*T_fine_coarse + 3*T_fine_fine + J_index_B_fine;
    dof = 7;
    if (std::find(lo_dofs.begin(), lo_dofs.end(), dof) != lo_dofs.end())
      AssertThrow(abs(system_matrix(dof, dof) - A_ii_an)/A_ii_an < eps,
                  ExcMessage("wrong "+std::to_string(dof)+", "+std::to_string(dof)));
    // -----------------------------------------------------------------------
    fluid_solver.solve_pressure_system();
    // const int n_pressure_iter = fluid_solver.solve();
    // pcout << "Pressure solver " << n_pressure_iter << " steps" << std::endl;
  } // eom

} // end of namespace

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    dealii::deallog.depth_console (0);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    // std::string input_file_name = Parsers::parse_command_line(argc, argv);
    std::string input_file_name = SOURCE_DIR "/../data/sf-4x4.data";
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

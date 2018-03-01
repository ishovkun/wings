#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/tria.h>

// Custom modules
#include <Model.hpp>
#include <Reader.hpp>

// #include <Wellbore.hpp>
#include <SolverIMPES.hpp>
#include <FEFunction/FEFunction.hpp>
// #include <FEFunction/FEFunctionPVT.hpp>
#include <FluidEquations/FluidEquationsSaturation.hpp>

namespace Wings
{
  using namespace dealii;


  template <int dim>
  class Simulator
  {
  public:
    Simulator(std::string);
    // ~Simulator();
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
  Simulator<dim>::Simulator(std::string input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator),
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    model(mpi_communicator, pcout),
    input_file(input_file_name_)
    // ,computing_timer(mpi_communicator, pcout,
    //                 TimerOutput::summary, TimerOutput::wall_times)
  {}


  template <int dim>
  void Simulator<dim>::read_mesh()
  {
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(model.mesh_config.file.string());

    // typename GridIn<dim>::format format = gridin<dim>::ucd;
    // gridin.read(f, format);
    gridin.read_msh(f);
    GridTools::scale(model.units.length(), triangulation);
  }  // eom


  template <int dim>
  void Simulator<dim>::refine_mesh()
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
  void Simulator<dim>::run()
  {
    Parsers::Reader reader(pcout, model);
    reader.read_input(input_file, /* verbosity= */0);
    read_mesh();
    // refine_mesh();

    FluidEquations::FluidEquationsPressure<dim>
        cell_values(model), neighbor_values(model);
    FluidEquations::FluidEquationsSaturation<dim> cell_values_saturation(model);
    FluidSolvers::SolverIMPES<dim> fluid_solver(mpi_communicator,
                                                triangulation,
                                                model, pcout,
                                                cell_values, neighbor_values,
                                                cell_values_saturation);
    fluid_solver.setup_dofs();

    // initial values
    fluid_solver.solution = 0.2;
    fluid_solver.saturation_relevant[0] = fluid_solver.solution;
    fluid_solver.saturation_old[0] = fluid_solver.solution;
    fluid_solver.solution = 0.8;
    fluid_solver.saturation_relevant[1] = fluid_solver.solution;

    fluid_solver.solution = 6894760;
    fluid_solver.pressure_relevant = fluid_solver.solution;
    fluid_solver.pressure_old = fluid_solver.solution;

    // double time = 1;

    double time = 0;
    double time_step = model.min_time_step;
    model.update_well_controls(time);

    model.locate_wells(fluid_solver.get_dof_handler());
    // std::vector<TrilinosWrappers::MPI::Vector*> saturation_solution =
    //     {&satura};

    FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
        pressure_function(fluid_solver.get_dof_handler(),
                          fluid_solver.pressure_relevant);
    FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
        saturation_function(fluid_solver.get_dof_handler(),
                            fluid_solver.saturation_relevant);

    const double p = 6894760;
    // test pvt
    std::vector<double>      pvt_values_water(4);
    model.get_pvt_water(p, pvt_values_water);
    // const double Bw = pvt_values_water[0];
    // const double Cw = pvt_values_water[1];
    // const double muw = pvt_values_water[2];
    // std::cout << "mu_w " << muw << std::endl;
    // std::cout << "B_w " << Bw << std::endl;
    // std::cout << "c_w " << Cw << std::endl;

    std::vector<double>      pvt_values_oil(4);
    model.get_pvt_oil(p, pvt_values_oil);
    // const double Bo = pvt_values_oil[0];
    // const double Co = pvt_values_oil[1];
    // const double muo = pvt_values_oil[2];
    // std::cout << "mu_o " << muo << std::endl;
    // std::cout << "B_o " << Bo << std::endl;
    // std::cout << "c_o " << Co << std::endl;

    {
      // test rel_perm
      Vector<double> saturation(2);
      std::vector<double> rel_perm(2);
      saturation[0] = 0.2;
      saturation[1] = 1-saturation[0];
      model.get_relative_permeability(saturation, rel_perm);
      // std::cout << "kw " << rel_perm[0] << std::endl;
      // std::cout << "ko " << rel_perm[1] << std::endl;

    }


    model.update_well_productivities(pressure_function, saturation_function);

    fluid_solver.assemble_pressure_system(time_step);

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


    const auto & system_matrix = fluid_solver.get_system_matrix();
    // system_matrix.print(std::cout, true);

    const double ft = model.units.length();
    const double psi = model.units.pressure();
    const double day = model.units.time();
    const double t_factor = 6.33e-3;
    const double tol = DefaultValues::small_number_balhoff;

    const double D_entry = 307.84*ft*ft*ft/psi/day;
    const double J_entry = 93361. * t_factor*ft*ft*ft/psi/day;
    const double Tx = 36000 *t_factor*ft*ft*ft/psi/day;
    const double Ty = 144000 *t_factor*ft*ft*ft/psi/day;
    const double Q1 = -2000*model.units.us_oil_barrel/day;
    const double Q2 = +3000*model.units.us_oil_barrel/day;
    const double Q8 = J_entry * 800.*psi;

    double m = 0; // numerical
    double a = 0;  // analytical
    int dof = 0;
    // double dof1 = 0 , dof2 = 0;

    // MATRIX
    // a11
    dof = 0;
    a = D_entry + Tx + Ty;
    m = system_matrix(dof, dof);
    if (Math::relative_difference(m,a) > tol)
    {
      std::cout << "A_an(" << dof<< "," << dof<<") = " << a << std::endl;;
      std::cout << "A(" << dof<< "," << dof<<") = " << m << std::endl;;
      std::cout << Math::relative_difference(m, a) << std::endl;
    }
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in A("+std::to_string(dof) +
                           ", "+std::to_string(dof)+")"));
    // a22
    dof = 1;
    a = D_entry + Tx + 2*Ty;
    m = system_matrix(dof, dof);
    if (Math::relative_difference(m,a) > tol)
    {
      std::cout << "A_an(" << dof<< "," << dof<<") = " << a << std::endl;;
      std::cout << "A(" << dof<< "," << dof<<") = " << m << std::endl;;
      std::cout << Math::relative_difference(m, a) << std::endl;
    }
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in A("+std::to_string(dof) +
                           ", "+std::to_string(dof)+")"));
    // a33
    dof = 2;
    a = D_entry + Tx + Ty;
    m = system_matrix(dof, dof);
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in A("+std::to_string(dof) +
                           ", "+std::to_string(dof)+")"));
    // a44
    dof = 3;
    a = D_entry + 2*Tx + Ty;
    m = system_matrix(dof, dof);
    // std::cout << "a11 = " << a << std::endl;;
    // std::cout << "m11 = " << m << std::endl;;
    // std::cout << relative_difference(m, a) << std::endl;
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in A("+std::to_string(dof) +
                           ", "+std::to_string(dof)+")"));
    // a55
    dof = 4;
    a = D_entry + 2*Tx + 2*Ty;
    m = system_matrix(dof, dof);
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in A("+std::to_string(dof) +
                           ", "+std::to_string(dof)+")"));
    // a99
    dof = 8;
    a = D_entry + Tx + Ty + J_entry;
    m = system_matrix(dof, dof);
    // std::cout << "a11 = " << a << std::endl;;
    // std::cout << "m11 = " << m << std::endl;;
    // std::cout << relative_difference(m, a) << std::endl;
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in A("+std::to_string(dof) +
                           ", "+std::to_string(dof)+")"));

    // RHS VECTOR
    const auto & rhs_vector = fluid_solver.get_rhs_vector();
    // system_matrix.print(std::cout, true);
    // rhs_vector.print(std::cout, 3, true, false);

    dof = 0;
    a = D_entry*fluid_solver.pressure_old[dof] + Q1;
    m = rhs_vector(dof);
    if (Math::relative_difference(m,a) > tol)
    {
      pcout << "analyt " << a << "\t" << std::endl;
      pcout << "numerical " << m << "\t" << std::endl;
      pcout << "rel dirr " << Math::relative_difference(m, a) << "\t" << std::endl;
    }
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
    dof = 1;
    a = D_entry*fluid_solver.pressure_old[dof];
    m = rhs_vector(dof);
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
    dof = 4;
    a = D_entry*fluid_solver.pressure_old[dof] + Q2;
    m = rhs_vector(dof);
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
    dof = 8;
    a = D_entry*fluid_solver.pressure_old[dof] + Q8;
    // a = Q8;
    m = rhs_vector(dof);
    // std::cout << "b_an(" << dof<< ") = " << a << std::endl;;
    // std::cout << "b(" << dof<< ") = " << m << std::endl;;
    // std::cout << Math::relative_difference(m, a) << std::endl;
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));

    fluid_solver.solve_pressure_system();
    fluid_solver.pressure_relevant = fluid_solver.solution;


    if (model.fluid_model != Model::FluidModelType::Liquid)
    {
      // saturation_solver.solve(cell_values_saturation,
      //                         neighbor_values_pressure,
      //                         time_step,
      //                         fluid_solver.relevant_solution,
      //                         fluid_solver.old_solution);
      fluid_solver.solve_saturation_system(time_step);
      // fluid_solver.saturation_relevant_solution[0] =
      //     fluid_solver.solution;
      // fluid_solver.saturation_relevant_solution[1] =
      //     fluid_solver.saturation_solution[1];
      fluid_solver.saturation_relevant[0] = fluid_solver.solution;
      fluid_solver.saturation_old[0] = fluid_solver.solution;
      fluid_solver.saturation_relevant[1] = 1.0;
      fluid_solver.saturation_relevant[1] -= fluid_solver.solution;
      //     fluid_solver.saturation_solution[1];
    }

    // test pressure solution
    // fluid_solver.solution.print(std::cout, 3, true, false);

    //                          0     3   6     1    4    7   2    5    8
    // P_an_balhoff = {984, 990, 972, 990, 993, 958, 991, 984, 921};
    // cell order is different in wings
    std::vector<double> P_an = {984, 991, 990, 993, 993, 984, 972, 958, 921};
    for (auto & value : P_an)
      value = value*psi;

    for (int dof =0; dof<9; ++dof)
    {
      a = P_an[dof];
      m = fluid_solver.pressure_relevant(dof);
      if (Math::relative_difference(m, a) > tol)
      {
        std::cout << "first time step pressure solution" << std::endl;
        std::cout << "P_an(" << dof<< ") = " << a << std::endl;;
        std::cout << "P_num(" << dof<< ") = " << m << std::endl;;
      }
      AssertThrow(Math::relative_difference(m, a) < tol,
                  ExcMessage("Wrong entry in P("+std::to_string(dof) + ")"));
    }

    // check saturation solution
    // saturation_solver.solution[0].print(std::cout, 4, true, false);
    std::vector<double> S_an = {0.2, 0.2, 0.2, 0.2, 0.2004, 0.2, 0.2, 0.2, 0.2001};
    for (int dof =0; dof<9; ++dof)
    {
      a = S_an[dof];
      m = fluid_solver.saturation_relevant[0](dof);
      if (Math::relative_difference(m, a) > tol)
      {
        std::cout << "first time step saturation solution" << std::endl;
        std::cout << "S_an(" << dof<< ") = " << a << std::endl;;
        std::cout << "S_num(" << dof<< ") = " << m << std::endl;;
      }
      AssertThrow(Math::relative_difference(m, a) < tol,
                  ExcMessage("Wrong entry in Sw("+std::to_string(dof) + ")"));
    }


    // Second time step
    model.update_well_productivities(pressure_function, saturation_function);

    fluid_solver.pressure_old = fluid_solver.pressure_relevant;
    fluid_solver.assemble_pressure_system(time_step);

    fluid_solver.solve_pressure_system();
    fluid_solver.pressure_relevant = fluid_solver.solution;

    fluid_solver.solve_saturation_system(time_step);
    fluid_solver.saturation_relevant[0] =
        fluid_solver.solution;

    // new part
    const double D2_0 = 307.8380*ft*ft*ft/psi/day;
    const double D2_4 = 307.7886*ft*ft*ft/psi/day;

    // a11
    dof = 0;
    a = D2_0 + 1.7999*1e5*t_factor*ft*ft*ft/psi/day;
    m = system_matrix(dof, dof);
    if (Math::relative_difference(m,a) > tol)
    {
      std::cout << "A_an(" << dof<< "," << dof<<") = " << a << std::endl;;
      std::cout << "A(" << dof<< "," << dof<<") = " << m << std::endl;;
      std::cout << Math::relative_difference(m, a) << std::endl;
    }
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in A("+std::to_string(dof) +
                           ", "+std::to_string(dof)+")"));
    // a44
    dof = 4;
    a = D2_4 + 3.5926*1e5*t_factor*ft*ft*ft/psi/day;
    m = system_matrix(dof, dof);
    if (Math::relative_difference(m,a) > tol)
    {
      std::cout << "A_an(" << dof<< "," << dof<<") = " << a << std::endl;;
      std::cout << "A(" << dof<< "," << dof<<") = " << m << std::endl;;
      std::cout << Math::relative_difference(m, a) << std::endl;
    }
    AssertThrow(Math::relative_difference(m, a) < tol,
                ExcMessage("Wrong entry in A("+std::to_string(dof) +
                           ", "+std::to_string(dof)+")"));

    // end new part

    //                 0     3   6     1    4    7   2    5    8
    // P_an_balhoff = {970, 974, 947, 976, 977, 932, 978, 968, 896};
    P_an = {970, 976, 978, 974, 977, 968, 947, 932, 896};
    for (auto & value : P_an)
      value = value*psi;

    for (int dof =0; dof<9; ++dof)
    {
      a = P_an[dof];
      m = fluid_solver.pressure_relevant(dof);
      if (Math::relative_difference(m, a) > tol)
      {
        std::cout << "Second time step pressure" << std::endl;
        std::cout << "P_an(" << dof<< ") = " << a << std::endl;;
        std::cout << "P_num(" << dof<< ") = " << m << std::endl;;
      }
      AssertThrow(Math::relative_difference(m, a) < tol,
                  ExcMessage("Wrong entry in P("+std::to_string(dof) + ")"));
    }


    // check saturation solution
    // saturation_solver.solution[0].print(std::cout, 4, true, false);

    //           0     3      6     1    4       7       2    5    8
    // S_balhoff = {0.2, 0.2, 0.2001. 0.2, 0.2008, 0.2001, 0.2, 0.2, 0.2001};
    S_an = {0.2, 0.2, 0.2, 0.2, 0.2008, 0.2, 0.2001, 0.2001, 0.2001};
    for (int dof =0; dof<9; ++dof)
    {
      a = S_an[dof];
      m = fluid_solver.saturation_relevant[0](dof);
      if (Math::relative_difference(m, a) > tol)
      {
        std::cout << "second time step saturation solution" << std::endl;
        std::cout << "S_an(" << dof<< ") = " << a << std::endl;;
        std::cout << "S_num(" << dof<< ") = " << m << std::endl;;
      }
      AssertThrow(Math::relative_difference(m, a) < tol,
                  ExcMessage("Wrong entry in Sw("+std::to_string(dof) + ")"));
    }

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
    std::string input_file_name = SOURCE_DIR "/../data/wo-3x3.data";
    Wings::Simulator<3> problem(input_file_name);
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

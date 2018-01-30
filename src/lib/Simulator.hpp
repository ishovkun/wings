#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/grid/grid_generator.h> // to create mesh
#include <deal.II/grid/grid_out.h>

// Custom modules
#include <Model.hpp>
#include <Reader.hpp>

// #include <Wellbore.hpp>
#include <PressureSolver.hpp>
#include <SaturationSolver.hpp>
#include <FEFunction/FEFunction.hpp>
// #include <FEFunction/FEFunctionPVT.hpp>



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
  void create_mesh();
  void run();

 private:
  void refine_mesh();
  void field_report(const double time_step,
                    const unsigned int time_step_number,
                    const FluidSolvers::SaturationSolver<dim> &saturation_solver);

  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> triangulation;
  ConditionalOStream                        pcout;
  Model::Model<dim>                         model;
  FluidSolvers::PressureSolver<dim>         pressure_solver;
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
    pressure_solver(mpi_communicator, triangulation, model, pcout),
    input_file(input_file_name_)
    // ,computing_timer(mpi_communicator, pcout,
    //                 TimerOutput::summary, TimerOutput::wall_times)
{}



template <int dim>
void Simulator<dim>::create_mesh()
{
  // make grid with 102x1x1 elements,
  // hx = hy = hz = h = 25 ft
  std::vector<unsigned int > repetitions = {102, 1, 1};
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            repetitions,
                                            Point<dim>(0,   -25, -12.5),
                                            Point<dim>(510, 25, +12.5));
  GridOut grid_out;
  // std::ofstream out ("bl-mesh.vtk ");
  // grid_out.write_vtk(triangulation, out);
  std::ofstream out("buckley_leverett.msh");
  grid_out.write_msh(triangulation, out);

  GridTools::scale(model.units.length(), triangulation);
}  // eom



template <int dim>
void Simulator<dim>::read_mesh()
{
  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(model.mesh_file.string());

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
void
Simulator<dim>::
field_report(const double time,
             const unsigned int time_step_number,
             const FluidSolvers::SaturationSolver<dim> &saturation_solver)
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(pressure_solver.get_dof_handler());
  data_out.add_data_vector(pressure_solver.relevant_solution, "pressure",
                           DataOut<dim>::type_dof_data);
  data_out.add_data_vector(saturation_solver.relevant_solution[0], "Sw",
                           DataOut<dim>::type_dof_data);
  data_out.build_patches();

  std::ostringstream filename;
  // filename << "./" << data.output_directory
  //          << "/"  << data.data_set_name
  //          << Utilities::int_to_string(time_step_number, 4)
  //          << ".vtk";
  filename << "solution."  << time_step_number << ".vtk";
  std::ofstream output(filename.str().c_str());
  data_out.write_vtk(output);
}




template <int dim>
void Simulator<dim>::run()
{
  Parsers::Reader reader(pcout, model);
  reader.read_input(input_file, /* verbosity= */0);

  // read_mesh();
  create_mesh();

  FluidSolvers::SaturationSolver<dim>
      saturation_solver(mpi_communicator,
                        pressure_solver.get_dof_handler(),
                        model, pcout);

  pressure_solver.setup_dofs();

  // tolerance for parameter check
  const double tol_p  = DefaultValues::small_number;
  { // test permeability function
    // first and last points have 1000 md perm, others - 50 md
    const double md = model.units.permeability();
    const double ft = model.units.length();
    const double k1 = model.get_permeability->value(Point<dim>(0.0, 0.0, 0.0));
    AssertThrow(Math::relative_difference(k1, 1000*md) < tol_p,
                ExcMessage("bug in perm function"));
    // std::cout << k1/md << std::endl;
    const double k2 = model.get_permeability->value(Point<dim>(6*ft, 0.0, 0.0));
    AssertThrow(Math::relative_difference(k2, 50*md) < tol_p,
                ExcMessage("bug in perm function"));
    // std::cout << k2/md << std::endl;
    const double k3 = model.get_permeability->value(Point<dim>(510*ft, 0.0, 0.0));
    AssertThrow(Math::relative_difference(k3, 1000*md) < tol_p,
                ExcMessage("bug in perm function"));
    // std::cout << k3/md << std::endl;
  }

  // if multiphase
  saturation_solver.setup_dofs(pressure_solver.locally_owned_dofs,
                               pressure_solver.locally_relevant_dofs);


  // initial values
  for (unsigned int i=0; i<saturation_solver.solution[0].size(); ++i)
  {
    saturation_solver.solution[0][i] =0.2;
    pressure_solver.solution = 1000*model.units.pressure();
  }
  saturation_solver.relevant_solution[0] = saturation_solver.solution[0];
  pressure_solver.relevant_solution = pressure_solver.solution;

  model.locate_wells(pressure_solver.get_dof_handler());

  CellValues::CellValuesBase<dim> cell_values_pressure(model),
                                  neighbor_values_pressure(model);
  CellValues::CellValuesSaturation<dim> cell_values_saturation(model);

  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      pressure_function(pressure_solver.get_dof_handler(),
                        pressure_solver.relevant_solution);
  FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
      saturation_function(pressure_solver.get_dof_handler(),
                          saturation_solver.relevant_solution);

  double time = 0;
  double time_step = model.min_time_step;
  unsigned int time_step_number = 0;

  while(time <= model.t_max)
  {
    time += time_step;
    pressure_solver.old_solution = pressure_solver.solution;

    pcout << "time " << time/model.units.time() << std::endl;
    // pcout << "tmax " << model.t_max/model.units.time() << std::endl;
    model.update_well_controls(time);
    model.update_well_productivities(pressure_function, saturation_function);

    { // solve for pressure
      pressure_solver.assemble_system(cell_values_pressure, neighbor_values_pressure,
                                      time_step,
                                      saturation_solver.relevant_solution);
      pressure_solver.solve();
      pressure_solver.relevant_solution = pressure_solver.solution;
    }

    { // solve for saturation
      saturation_solver.solve(cell_values_saturation,
                              neighbor_values_pressure,
                              time_step,
                              pressure_solver.relevant_solution,
                              pressure_solver.old_solution);
      saturation_solver.relevant_solution[0] = saturation_solver.solution[0];
      saturation_solver.relevant_solution[1] = saturation_solver.solution[1];
    }


    {  // construct dimensionless parameters and get saturation profile
      auto & injector = model.wells[0];
      const double q_rate = injector.get_control().value;
      const double ft = model.units.length();
      const double area = 25*50*ft*ft;
      const double length = 510*ft;
      const double phi = model.get_porosity->value(Point<dim>(0,0,0));
      const double dimensionless_time =
          q_rate*time/area/length/phi;
      pcout << "td = " << dimensionless_time << std::endl;
    }

    field_report(time, time_step_number, saturation_solver);

    time_step_number++;
  }

  // model.update_well_controls(time);

  // CellValues::CellValuesBase<dim> cell_values_pressure(model),
  //                                 neighbor_values_pressure(model);
  // // pointer to cell values that are gonna be used
  // CellValues::CellValuesBase<dim>* p_cell_values = &cell_values_pressure;
  // CellValues::CellValuesBase<dim>* p_neighbor_values = &neighbor_values_pressure;
  // // CellValues::CellValuesBase<dim>* p_cell_values = NULL;
  // // CellValues::CellValuesBase<dim>* p_neighbor_values = NULL;
  // // if (model.type == Model::ModelType::SingleLiquid)
  // // {
  // //   p_cell_values = &cell_values_sf;
  // //   p_neighbor_values = &neighbor_values_sf;
  // // }
  // // else
  // // {

  // //   p_cell_values = &cell_values_mp;
  // //   p_neighbor_values = &neighbor_values_mp;
  // // }

  // model.locate_wells(pressure_solver.get_dof_handler());
  // // std::vector<TrilinosWrappers::MPI::Vector*> saturation_solution =
  // //     {&satura};

  // FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
  //     saturation_function(pressure_solver.get_dof_handler(),
  //                         saturation_solver.relevant_solution);
  // // {// test saturation values
  // //   Vector<double> tmp(2);
  // //   saturation_function.vector_value(Point<dim>{0,0,0}, tmp);
  // //   // pcout << "Sw " << tmp[0] << std::endl;
  // // }

  // FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
  //     pressure_function(pressure_solver.get_dof_handler(),
  //                       pressure_solver.relevant_solution);

  // const double p = 6894760;
  // // test pvt
  // std::vector<double>      pvt_values_water(4);
  // model.get_pvt_water(p, pvt_values_water);
  // // const double Bw = pvt_values_water[0];
  // // const double Cw = pvt_values_water[1];
  // // const double muw = pvt_values_water[2];
  // // std::cout << "mu_w " << muw << std::endl;
  // // std::cout << "B_w " << Bw << std::endl;
  // // std::cout << "c_w " << Cw << std::endl;

  // std::vector<double>      pvt_values_oil(4);
  // model.get_pvt_oil(p, pvt_values_oil);
  // // const double Bo = pvt_values_oil[0];
  // // const double Co = pvt_values_oil[1];
  // // const double muo = pvt_values_oil[2];
  // // std::cout << "mu_o " << muo << std::endl;
  // // std::cout << "B_o " << Bo << std::endl;
  // // std::cout << "c_o " << Co << std::endl;

  // {
  //   // test rel_perm
  //   Vector<double> saturation(2);
  //   std::vector<double> rel_perm(2);
  //   saturation[0] = 0.2;
  //   saturation[1] = 1-saturation[0];
  //   model.get_relative_permeability(saturation, rel_perm);
  //   // std::cout << "kw " << rel_perm[0] << std::endl;
  //   // std::cout << "ko " << rel_perm[1] << std::endl;

  // }


  // model.update_well_productivities(pressure_function, saturation_function);

  // pressure_solver.assemble_system(*p_cell_values, *p_neighbor_values,
  //                                 time_step,
  //                                 saturation_solver.relevant_solution);

  // // for (auto & id : model.get_well_ids())
  // // {
  // //   std::cout << "well_id " << id << std::endl;
  // //   auto & well = model.wells[id];

  // //   std::cout << "Real locations"  << std::endl;
  // //   for (auto & loc : well.get_locations())
  // //     std::cout << loc << std::endl;

  // //   std::cout << "Assigned locations"  << std::endl;
  // //   for (auto & cell : well.get_cells())
  // //     std::cout << cell->center() << std::endl;

  // //   std::cout << std::endl;
  // // }


  // const auto & system_matrix = pressure_solver.get_system_matrix();
  // // system_matrix.print(std::cout, true);

  // const double ft = model.units.length();
  // const double psi = model.units.pressure();
  // const double day = model.units.time();
  // const double t_factor = 6.33e-3;
  // const double tol = DefaultValues::small_number_balhoff;

  // const double D_entry = 307.84*ft*ft*ft/psi/day;
  // const double J_entry = 93361. * t_factor*ft*ft*ft/psi/day;
  // const double Tx = 36000 *t_factor*ft*ft*ft/psi/day;
  // const double Ty = 144000 *t_factor*ft*ft*ft/psi/day;
  // const double Q1 = -2000*model.units.us_oil_barrel/day;
  // const double Q2 = +3000*model.units.us_oil_barrel/day;
  // const double Q8 = J_entry * 800.*psi;

  // double m = 0; // numerical
  // double a = 0;  // analytical
  // int dof = 0;
  // // double dof1 = 0 , dof2 = 0;

  // // a11
  // dof = 0;
  // a = D_entry + Tx + Ty;
  // m = system_matrix(dof, dof);
  // if (Math::relative_difference(m,a) > tol)
  // {
  //   std::cout << "A_an(" << dof<< "," << dof<<") = " << a << std::endl;;
  //   std::cout << "A(" << dof<< "," << dof<<") = " << m << std::endl;;
  //   std::cout << Math::relative_difference(m, a) << std::endl;
  // }
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in A("+std::to_string(dof) +
  //                        ", "+std::to_string(dof)+")"));
  // // a22
  // dof = 1;
  // a = D_entry + Tx + 2*Ty;
  // m = system_matrix(dof, dof);
  // if (Math::relative_difference(m,a) > tol)
  // {
  //   std::cout << "A_an(" << dof<< "," << dof<<") = " << a << std::endl;;
  //   std::cout << "A(" << dof<< "," << dof<<") = " << m << std::endl;;
  //   std::cout << Math::relative_difference(m, a) << std::endl;
  // }
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in A("+std::to_string(dof) +
  //                        ", "+std::to_string(dof)+")"));
  // // a33
  // dof = 2;
  // a = D_entry + Tx + Ty;
  // m = system_matrix(dof, dof);
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in A("+std::to_string(dof) +
  //                        ", "+std::to_string(dof)+")"));
  // // a44
  // dof = 3;
  // a = D_entry + 2*Tx + Ty;
  // m = system_matrix(dof, dof);
  // // std::cout << "a11 = " << a << std::endl;;
  // // std::cout << "m11 = " << m << std::endl;;
  // // std::cout << relative_difference(m, a) << std::endl;
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in A("+std::to_string(dof) +
  //                        ", "+std::to_string(dof)+")"));
  // // a55
  // dof = 4;
  // a = D_entry + 2*Tx + 2*Ty;
  // m = system_matrix(dof, dof);
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in A("+std::to_string(dof) +
  //                        ", "+std::to_string(dof)+")"));
  // // a99
  // dof = 8;
  // a = D_entry + Tx + Ty + J_entry;
  // m = system_matrix(dof, dof);
  // // std::cout << "a11 = " << a << std::endl;;
  // // std::cout << "m11 = " << m << std::endl;;
  // // std::cout << relative_difference(m, a) << std::endl;
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in A("+std::to_string(dof) +
  //                        ", "+std::to_string(dof)+")"));

  // // RHS VECTOR
  // const auto & rhs_vector = pressure_solver.get_rhs_vector();
  // // system_matrix.print(std::cout, true);
  // // rhs_vector.print(std::cout, 3, true, false);

  // dof = 0;
  // a = D_entry*pressure_solver.old_solution[dof] + Q1;
  // m = rhs_vector(dof);
  // if (Math::relative_difference(m,a) > tol)
  // {
  //   pcout << "analyt " << a << "\t" << std::endl;
  //   pcout << "numerical " << m << "\t" << std::endl;
  //   pcout << "rel dirr " << Math::relative_difference(m, a) << "\t" << std::endl;
  // }
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
  // dof = 1;
  // a = D_entry*pressure_solver.old_solution[dof];
  // m = rhs_vector(dof);
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
  // dof = 4;
  // a = D_entry*pressure_solver.old_solution[dof] + Q2;
  // m = rhs_vector(dof);
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
  // dof = 8;
  // a = D_entry*pressure_solver.old_solution[dof] + Q8;
  // // a = Q8;
  // m = rhs_vector(dof);
  // // std::cout << "b_an(" << dof<< ") = " << a << std::endl;;
  // // std::cout << "b(" << dof<< ") = " << m << std::endl;;
  // // std::cout << Math::relative_difference(m, a) << std::endl;
  // AssertThrow(Math::relative_difference(m, a) < tol,
  //             ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));

  // pressure_solver.solve();
  // pressure_solver.relevant_solution = pressure_solver.solution;


  // CellValues::CellValuesSaturation<dim> cell_values_saturation(model);

  // if (model.type != Model::ModelType::SingleLiquid)
  // {
  //   saturation_solver.solve(cell_values_saturation,
  //                           neighbor_values_pressure,
  //                           time_step,
  //                           pressure_solver.relevant_solution,
  //                           pressure_solver.old_solution);
  //   saturation_solver.relevant_solution[0] = saturation_solver.solution[0];
  //   saturation_solver.relevant_solution[1] = saturation_solver.solution[1];
  // }

  // // test pressure solution
  // // pressure_solver.solution.print(std::cout, 3, true, false);

  // //                          0     3   6     1    4    7   2    5    8
  // // P_an_balhoff = {984, 990, 972, 990, 993, 958, 991, 984, 921};
  // // cell order is different in wings
  // std::vector<double> P_an = {984, 991, 990, 993, 993, 984, 972, 958, 921};
  // for (auto & value : P_an)
  //   value = value*psi;

  // for (int dof =0; dof<9; ++dof)
  // {
  //   a = P_an[dof];
  //   m = pressure_solver.solution(dof);
  //   if (Math::relative_difference(m, a) > tol)
  //   {
  //     std::cout << "b_an(" << dof<< ") = " << a << std::endl;;
  //     std::cout << "b(" << dof<< ") = " << m << std::endl;;
  //   }
  //   AssertThrow(Math::relative_difference(m, a) < tol,
  //               ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
  // }

  // // check saturation solution
  // // saturation_solver.solution[0].print(std::cout, 4, true, false);
  // std::vector<double> S_an = {0.2, 0.2, 0.2, 0.2, 0.2004, 0.2, 0.2, 0.2, 0.2001};
  // for (int dof =0; dof<9; ++dof)
  // {
  //   a = S_an[dof];
  //   m = saturation_solver.solution[0](dof);
  //   if (Math::relative_difference(m, a) > tol)
  //   {
  //     std::cout << "b_an(" << dof<< ") = " << a << std::endl;;
  //     std::cout << "b(" << dof<< ") = " << m << std::endl;;
  //   }
  //   AssertThrow(Math::relative_difference(m, a) < tol,
  //               ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
  // }


  // // Second time step
  // model.update_well_productivities(pressure_function, saturation_function);

  // pressure_solver.old_solution = pressure_solver.solution;
  // pressure_solver.assemble_system(*p_cell_values, *p_neighbor_values,
  //                                 time_step,
  //                                 saturation_solver.relevant_solution);

  // pressure_solver.solve();
  // pressure_solver.relevant_solution = pressure_solver.solution;
  // if (model.type != Model::ModelType::SingleLiquid)
  // {
  //   saturation_solver.solve(cell_values_saturation,
  //                           neighbor_values_pressure,
  //                           time_step,
  //                           pressure_solver.relevant_solution,
  //                           pressure_solver.old_solution);
  //   saturation_solver.relevant_solution[0] = saturation_solver.solution[0];
  //   saturation_solver.relevant_solution[1] = saturation_solver.solution[1];
  // }

  // //                 0     3   6     1    4    7   2    5    8
  // // P_an_balhoff = {970, 974, 947, 976, 977, 932, 978, 968, 896};
  // P_an = {970, 976, 978, 974, 977, 968, 947, 932, 896};
  // for (auto & value : P_an)
  //   value = value*psi;

  // for (int dof =0; dof<9; ++dof)
  // {
  //   a = P_an[dof];
  //   m = pressure_solver.solution(dof);
  //   if (Math::relative_difference(m, a) > tol)
  //   {
  //     std::cout << "b_an(" << dof<< ") = " << a << std::endl;;
  //     std::cout << "b(" << dof<< ") = " << m << std::endl;;
  //   }
  //   AssertThrow(Math::relative_difference(m, a) < tol,
  //               ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
  // }


  // // check saturation solution
  // // saturation_solver.solution[0].print(std::cout, 4, true, false);

  // //           0     3      6     1    4       7       2    5    8
  // // S_balhoff = {0.2, 0.2, 0.2001. 0.2, 0.2008, 0.2001, 0.2, 0.2, 0.2001};
  // S_an = {0.2, 0.2, 0.2, 0.2, 0.2008, 0.2, 0.2001, 0.2001, 0.2001};
  // for (int dof =0; dof<9; ++dof)
  // {
  //   a = S_an[dof];
  //   m = saturation_solver.solution[0](dof);
  //   if (Math::relative_difference(m, a) > tol)
  //   {
  //     std::cout << "b_an(" << dof<< ") = " << a << std::endl;;
  //     std::cout << "b(" << dof<< ") = " << m << std::endl;;
  //   }
  //   AssertThrow(Math::relative_difference(m, a) < tol,
  //               ExcMessage("Wrong entry in b("+std::to_string(dof) + ")"));
  // }

} // eom

} // end of namespace

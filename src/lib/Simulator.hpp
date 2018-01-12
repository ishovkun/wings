#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/distributed/tria.h>

// Custom modules
#include <Model.hpp>
#include <Reader.hpp>

// #include <Wellbore.hpp>
#include <PressureSolver.hpp>
#include <SaturationSolverExplicit.hpp>
#include <FEFunction.hpp>
// #include <CellValues.hpp>

#include <ExtraFEData.hpp>

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
  void Simulator<dim>::read_mesh()
  {
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f(model.mesh_file.string());

    // typename GridIn<dim>::format format = gridin<dim>::ucd;
    // gridin.read(f, format);
    gridin.read_msh(f);
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

    FluidSolvers::SaturationSolverExplicit<dim>
      saturation_solver(2, mpi_communicator,
                        pressure_solver.get_dof_handler(),
                        model, pcout);

    pressure_solver.setup_dofs();
    // if multiphase
    saturation_solver.setup_dofs(pressure_solver.locally_owned_dofs,
                                 pressure_solver.locally_relevant_dofs);

    // double time = 1;
    const DoFHandler<dim> & pressure_dof_handler =
      pressure_solver.get_dof_handler();
    // std::vector< std::vector<TrilinosWrappers::MPI::Vector*> > extra_vectors(1);
    // // std::vector< TrilinosWrappers::MPI::Vector* >
    // //   extra_vectors;
    // extra_vectors[0].push_back(&(saturation_solver.relevant_solution[0]));
    // const int dim = 3;
    ExtraFEData::ExtraFEData<dim> extra_data(std::vector<unsigned int> {{1}});
    std::vector< const DoFHandler<dim>* > handlers =
        {&pressure_dof_handler};

    std::vector< std::vector<ExtraFEData::FEDerivativeOrder> > derivatives =
        {{ExtraFEData::FEDerivativeOrder::values}};

    std::vector< std::vector<TrilinosWrappers::MPI::Vector*> > extra_vectors
        = {{&(saturation_solver.relevant_solution[0])}};
    extra_data.set_data(handlers, extra_vectors, derivatives);

    // QGauss<dim>  quadrature_formula(1);
    // extra_data.make_fe_values(quadrature_formula);

    double time_step = model.min_time_step;
    CellValues::CellValuesBase<dim> cell_values_sf(model),
                                    neighbor_values_sf(model);
    CellValues::CellValuesMP<dim> cell_values_mp(model),
                                  neighbor_values_mp(model);
    // pointer to cell values that are gonna be used
    CellValues::CellValuesBase<dim>* p_cell_values = NULL;
    CellValues::CellValuesBase<dim>* p_neighbor_values = NULL;
    if (model.type == Model::ModelType::SingleLiquid)
    {
      p_cell_values = &cell_values_sf;
      p_neighbor_values = &neighbor_values_sf;
    }
    else
    {
      p_cell_values = &cell_values_mp;
      p_neighbor_values = &neighbor_values_mp;
    }

    model.locate_wells(pressure_solver.get_dof_handler());
    // std::vector<TrilinosWrappers::MPI::Vector*> saturation_solution =
    //     {&satura};

    FEFunction::FEFunction<dim,TrilinosWrappers::MPI::Vector>
        saturation_function(pressure_solver.get_dof_handler(),
                            saturation_solver.relevant_solution);
    {// test saturation values
      Vector<double> tmp(2);
      saturation_function.vector_value(Point<dim>{0,0,0}, tmp);
      pcout << "Sw " << tmp[0] << std::endl;
    }

    model.update_well_productivities(saturation_function);

    // pressure_solver.assemble_system(*p_cell_values, *p_neighbor_values,
    //                                 time_step, extra_data);
  } // eom

} // end of namespace

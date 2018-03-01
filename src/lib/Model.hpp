#pragma once

// #include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

// Custom modules
#include <Wellbore.hpp>
#include <Parsers.hpp>
#include <BitMap.hpp>
#include <Units.h>
#include <Tensors.hpp>
#include <LookupTable.hpp>
#include <RelativePermeability.hpp>


namespace Model
{
using namespace dealii;

enum FluidModelType {Liquid, Gas, DeadOil, WaterGas, BlackOil};

enum SolidModelType {Compressibility, Elasticity};

enum PVTType {Constant, Table, Correlation};

enum Phase {Water, Oleic, Gaseous};

enum FluidCouplingStrategy {None, FixedStressSplit};

enum LinearSolverType {Direct, CG, GMRES};

// whether to request to create mesh or read in .msh or abaqus formats
enum MeshType {Create, Msh, Abaqus};

// This is to extend to functions and correlations instead of tables
struct ModelConfig
{
  PVTType pvt_oil, pvt_water, pvt_gas;
};

struct MeshConfig
{
  MeshConfig() {n_cells.resize(3);}
  MeshType type;
  // number of cells in each direction
  std::vector<unsigned int>    n_cells;
  // two corner points of the parallelepiped mesh
  std::pair<Point<3>,Point<3>> points;
  boost::filesystem::path      file;
};


struct PVTValues
{
  double volume_factor, viscosity, compressibility;
};



template <int dim>
class Model
{

  // template<int dim> Model<dim>
  // friend class Parsers::Reader;
 public:
  Model(MPI_Comm           &mpi_communicator_,
        ConditionalOStream &pcout_);
  ~Model();
 private:
  MPI_Comm                               & mpi_communicator;
  ConditionalOStream                     & pcout;
 public:
  // void read_input(const std::string&,
  //                 const int verbosity_=0);
  // void print_input();

  // Functions of a coordinate
  Function<dim> * get_young_modulus,
                * get_poisson_ratio,
                * get_permeability,
                * get_porosity;

  // adding data
  void set_fluid_model(const FluidModelType &type);
  void set_solid_model(const SolidModelType &type);
  void set_pvt_water(Interpolation::LookupTable &table);
  void set_pvt_oil(Interpolation::LookupTable &table);
  void set_pvt_gas(Interpolation::LookupTable &table);
  void set_rel_perm(const double Sw_crit,
                    const double So_rw,
                    const double k_rw0,
                    const double k_ro0,
                    const double nw,
                    const double no);
  // this method is for elasticity model only
  void set_biot_coefficient(const double x){biot_coefficient = x;}
  // this method is for Compressibility model only
  void set_rock_compressibility(const double x) {rock_compressibility_constant = x;}
  void set_density_sc_w(const double x) {density_sc_w_constant = x;}
  void set_density_sc_o(const double x) {density_sc_o_constant = x;}
  void add_well(const std::string name,
                const double radius,
                const std::vector< Point<dim> > &locations);
  void set_solid_dirichlet_boundary_conditions(const std::vector<int>    & labels,
                                               const std::vector<int>    & components,
                                               const std::vector<double> & values);
  void set_solid_neumann_boundary_conditions(const std::vector<int>    & labels,
                                             const std::vector<int>    & components,
                                             const std::vector<double> & values);
  void set_fluid_linear_solver(const LinearSolverType & solver_type);
  void set_solid_linear_solver(const LinearSolverType & solver_type);
  // querying data
  bool has_phase(const Phase &phase) const;
  unsigned int n_phases() const;
  double density_standard_conditions(const int phase) const;
  double density_sc_water() const;
  double density_sc_oil() const;
  double get_biot_coefficient() const;
  double gravity() const;
  // C_r = \partial poro / \partial p
  double get_rock_compressibility(const Point<dim> &p) const;
  void get_pvt_oil(const double        pressure,
                   std::vector<double> &dst) const;
  void get_pvt_water(const double        pressure,
                     std::vector<double> &dst) const;
  void get_pvt_gas(const double        pressure,
                   std::vector<double> &dst) const;
  // this function calls any of the three above allowing for
  // phase-agnostic FluidEquations
  void get_pvt(const double        pressure,
               const int           phase,
               std::vector<double> &dst) const;
  double get_time_step(const double time) const;
  std::vector<int> get_well_ids() const;
  void get_relative_permeability(Vector<double>      &saturation,
                                 std::vector<double> &dst) const;
  int get_well_id(const std::string& well_name) const;
  std::pair<double,double> get_saturation_limits(const unsigned int phase) const;

  const Interpolation::LookupTable &
  get_pvt_table_water() const {return pvt_table_water;}
  const Interpolation::LookupTable &
  get_pvt_table_oil() const {return pvt_table_oil;}
  double residual_saturation_water() const;
  double residual_saturation_oil() const;
  FluidCouplingStrategy coupling_strategy() const;

  // update methods
  void update_well_controls(const double time);
  void locate_wells(const DoFHandler<dim>& dof_handler);
  void update_well_productivities(const Function<dim> &get_pressure,
                                  const Function<dim> &get_saturation);

  void compute_runtime_parameters();
  const std::vector<const Interpolation::LookupTable*> get_pvt_tables() const;

  // ATTRIBUTES
 public:
  std::vector<int>           solid_dirichlet_labels;
  std::vector<int>           solid_dirichlet_components;
  std::vector<double>        solid_dirichlet_values;
  std::vector<int>           solid_neumann_labels;
  std::vector<int>           solid_neumann_components;
  std::vector<double>        solid_neumann_values;
  const unsigned int         n_pvt_water_columns = 5;
  const unsigned int         n_pvt_oil_columns = 5;
  const unsigned int         n_pvt_gas_columns = 5;
  int                        initial_refinement_level,
                             n_adaptive_steps;
  std::vector< std::pair< Point<dim>,Point<dim> > >
  local_prerefinement_region;
  Units::Units               units;
  std::vector<Wellbore<dim>> wells;
  Schedule::Schedule         schedule;
  double                     coupling_tolerance,
                             min_time_step,
                             t_max;
  int                        max_coupling_steps;

  FluidModelType             fluid_model;
  SolidModelType             solid_model;
  // this thing is to specify the types of pvts (table, correlation) and relperms
  ModelConfig                config; // not used anywhere
  LinearSolverType           linear_solver_solid,
                             linear_solver_fluid;
  MeshConfig                 mesh_config;
  MeshType                   mesh_type;
  double                     reference_pressure;
  double                     initial_saturation_water, initial_saturation_oil;

 protected:
  std::string                input_file_name;
  double                     density_sc_w_constant,
                             density_sc_o_constant,
                             porosity,
                             biot_coefficient,
                             rock_compressibility_constant;
  Interpolation::LookupTable pvt_table_water,
                             pvt_table_oil,
                             pvt_table_gas;
  RelativePermeability       rel_perm;
  std::vector<Phase>         phases;
 private:
  std::map<double, double>   timestep_table;
  std::map<std::string, int> well_ids;
};  // eom


template <int dim>
Model<dim>::Model(MPI_Comm           &mpi_communicator_,
                  ConditionalOStream &pcout_)
    :
    mpi_communicator(mpi_communicator_),
    pcout(pcout_),
    get_young_modulus(NULL),
    get_poisson_ratio(NULL),
    get_permeability(NULL),
    get_porosity(NULL)
{
  // declare_parameters();
  units.set_system(Units::si_units);
}  // eom



template <int dim>
Model<dim>::~Model()
{
  delete get_young_modulus;
  delete get_permeability;
  delete get_porosity;
}



template <int dim>
inline
double Model<dim>::density_standard_conditions(const int phase) const
{
  AssertThrow(phase < n_phases(), ExcDimensionMismatch(phase, n_phases()));
  if (phase == 0)
    return this->density_sc_w_constant;

  // phase > 0
  if (fluid_model == FluidModelType::DeadOil)
    return this->density_sc_o_constant;
  else if (fluid_model == FluidModelType::WaterGas)
    return this->density_sc_g_constant;
  else if (fluid_model == FluidModelType::BlackOil)
    if (phase == 1)
      return this->density_sc_o_constant;
    else
      return this->density_sc_g_constant;

  return 0;

}  // eom



template <int dim>
inline
double Model<dim>::density_sc_water() const
{
  return this->density_sc_w_constant;
}  // eom



template <int dim>
inline
double Model<dim>::density_sc_oil() const
{
  return this->density_sc_o_constant;
}  // eom



template <int dim>
inline
double Model<dim>::gravity() const
{
  return units.gravity();
}  // eom


// template <int dim>
// void Model<dim>::parse_time_stepping()
// {
//   // Parse time stepping table
//   std::vector<Point<2> > tmp =
//     Parsers::parse_point_list<2>(prm.get(keywords.time_stepping));
//   for (const auto &row : tmp)
//     this->timestep_table[row[0]] = row[1];
// } // eom


// template <int dim>
// double Model<dim>::get_time_step(const double time) const
// /* get value of the time step from the time-stepping table */
// {
//   double time_step = timestep_table.rbegin()->second;
//   for (const auto &it : timestep_table)
//     {
//       if (time >= it.first)
//         time_step = it.second;
//       else
//         break;
//     }

//   return time_step;
// }  // eom



template <int dim>
int Model<dim>::get_well_id(const std::string& well_name) const
{
  return well_ids.find(well_name)->second;
} // eom



template <int dim>
std::vector<int> Model<dim>::get_well_ids() const
{
  std::vector<int> result;
  for(auto & id : well_ids)
    result.push_back(id.second);
  return result;
} // eom



template <int dim>
const std::vector<const Interpolation::LookupTable*> Model<dim>::get_pvt_tables() const
{
  std::vector<const Interpolation::LookupTable*> pvt_tables;
  if (has_phase(Phase::Water))
    pvt_tables.push_back(&pvt_table_water);
  if (has_phase(Phase::Oleic))
    pvt_tables.push_back(&pvt_table_oil);
  // if (has_phase(const Model::Gas))
  //   pvt_tables.push_back(get_pvt_table_gas());

  return pvt_tables;
}  // eom


template <int dim>
void Model<dim>::add_well(const std::string name,
                          const double radius,
                          const std::vector< Point<dim> > &locations)
{
  Wellbore<dim> w(locations, radius, mpi_communicator,
                  *get_permeability, rel_perm, get_pvt_tables());
  this->wells.push_back(w);

  // check if well_id is in unique_well_ids and add if not
  if (well_ids.empty())
    well_ids[name] = 0;
  else
  {
    std::map<std::string, int>::iterator
        it = well_ids.begin(),
        end = well_ids.end();

    for (; it!=end; it++)
      AssertThrow(it->first != name, ExcMessage("Duplicates in wells"));

    const int id = well_ids.size();
    well_ids[name] = id;
  }
} // eom



template <int dim>
void Model<dim>::update_well_controls(const double time)
{
  for (unsigned int i=0; i<wells.size(); i++)
    wells[i].set_control(schedule.get_control(time, i));
} // eom



template <int dim>
void Model<dim>::locate_wells(const DoFHandler<dim>& dof_handler)
{
  for (unsigned int i=0; i<wells.size(); i++)
  {
    // std::cout << "well " << i << std::endl;
    wells[i].locate(dof_handler);
  }
} // eom



template <int dim>
void Model<dim>::update_well_productivities(const Function<dim> &get_pressure,
                                            const Function<dim> &get_saturation)
{
  for (auto & well : wells)
    well.update_productivity(get_pressure, get_saturation);
}  // eom



template <int dim>
void Model<dim>::set_pvt_water(Interpolation::LookupTable &table)
{
  pvt_table_water = table;
}  // eom



template <int dim>
void Model<dim>::set_pvt_oil(Interpolation::LookupTable &table)
{
  pvt_table_oil = table;
}  // eom



template <int dim>
void Model<dim>::get_pvt(const double        pressure,
                         const int           phase,
                         std::vector<double> &dst) const
{
  AssertThrow(phase < n_phases(), ExcMessage("Wrong phase"));

  if (fluid_model == FluidModelType::Liquid)
    get_pvt_water(pressure, dst);
  else if (fluid_model == DeadOil)
  {
    if (phase == 0)
      get_pvt_water(pressure, dst);
    else
      get_pvt_oil(pressure, dst);
  }
  else
    AssertThrow(false, ExcNotImplemented());
}



template <int dim>
void Model<dim>::get_pvt_water(const double        pressure,
                               std::vector<double> &dst) const
{
  AssertThrow(dst.size() == n_pvt_water_columns-1,
              ExcDimensionMismatch(dst.size(), n_pvt_water_columns-1));
  pvt_table_water.get_values(pressure, dst);
}  // eom


template <int dim>
void Model<dim>::get_pvt_oil(const double        pressure,
                             std::vector<double> &dst) const
{
  AssertThrow(dst.size() == n_pvt_oil_columns-1,
              ExcDimensionMismatch(dst.size(), n_pvt_oil_columns-1));
  pvt_table_oil.get_values(pressure, dst);
}  // eom



template <int dim>
void Model<dim>::set_solid_model(const SolidModelType & model_type)
{
  solid_model = model_type;
} // eom



template <int dim>
void Model<dim>::set_fluid_model(const FluidModelType & model_type)
{
  phases.clear();
  fluid_model = model_type;

  if (fluid_model == FluidModelType::Liquid)
    phases.push_back(Phase::Water);
  else if (fluid_model == DeadOil)
  {
    phases.push_back(Phase::Water);
    phases.push_back(Phase::Oleic);
  }
  else
    AssertThrow(false, ExcNotImplemented());
}



template <int dim>
inline
bool Model<dim>::has_phase(const Phase &phase) const
{
  for (const auto & p : phases)
    if (p == phase)
      return true;

  return false;
}  // eom



template <int dim>
inline
unsigned int Model<dim>::n_phases() const
{
  return phases.size();
}  // eom


template <int dim>
inline
void Model<dim>::set_rel_perm(const double Sw_crit,
                              const double So_rw,
                              const double k_rw0,
                              const double k_ro0,
                              const double nw,
                              const double no)
{
  rel_perm.set_data(Sw_crit, So_rw, k_rw0, k_ro0, nw, no);
}  // eom



template <int dim>
inline
void Model<dim>::get_relative_permeability(Vector<double>      &saturation,
                                           std::vector<double> &dst) const
{
  AssertThrow(dst.size() == n_phases(),
              ExcDimensionMismatch(dst.size(), n_phases()));

  if (fluid_model == Liquid)
    dst[0] = 1;
  else if (fluid_model == DeadOil)
    rel_perm.get_values(saturation, dst);
  else
    AssertThrow(false, ExcNotImplemented());
}



template <int dim>
inline
std::pair<double,double>
Model<dim>::get_saturation_limits(const unsigned int phase) const
{
  AssertThrow(phase < n_phases(), ExcMessage("Wrong phase index"));
  if (n_phases() == 1)
  {
    return std::make_pair(0.0, 1.0);
  }
  else if (fluid_model == FluidModelType::DeadOil)
  {
    if (phase == 0)
      return std::make_pair(rel_perm.Sw_crit, 1.0 - rel_perm.So_rw);
    else
      return std::make_pair(rel_perm.So_rw, 1.0 - rel_perm.Sw_crit);
  }
  else
    AssertThrow(false, ExcNotImplemented());

  // to supress warning
  return std::make_pair(0.0, 1.0);
}  // eom



template<int dim>
double
Model<dim>::get_rock_compressibility(const Point<dim> &p) const
{
  if (solid_model == SolidModelType::Compressibility)
  {
    return rock_compressibility_constant;
  }
  else if (solid_model == SolidModelType::Elasticity)
  {
    const double E = get_young_modulus->value(p);
    const double nu = get_poisson_ratio->value(p);
    const double bulk_modulus = E/3.0/(1.0-2.0*nu);
    const double phi = get_porosity->value(p);
    const double alpha = get_biot_coefficient();
    AssertThrow(alpha > phi /* || alpha == 0.0 */,
                ExcMessage("Biot coef should be > porosity"));
    const double rec_N = (alpha - phi) * (1.0 - alpha) / bulk_modulus;
    return rec_N;
  }
  else
  {
    AssertThrow(false, ExcNotImplemented());
  }
}  // end get_rock_compressibility



template<int dim>
void
Model<dim>::
set_solid_dirichlet_boundary_conditions(const std::vector<int>    & labels,
                                        const std::vector<int>    & components,
                                        const std::vector<double> & values)
{
  solid_dirichlet_labels = labels;
  solid_dirichlet_components = components;
  solid_dirichlet_values = values;
}  // end set_solid_dirichlet_boundary_conditions



template<int dim>
void
Model<dim>::
set_solid_neumann_boundary_conditions(const std::vector<int>    & labels,
                                      const std::vector<int>    & components,
                                      const std::vector<double> & values)
{
  solid_neumann_labels = labels;
  solid_neumann_components = components;
  solid_neumann_values = values;
}  // end set_solid_dirichlet_boundary_conditions



template<int dim>
double
Model<dim>::get_biot_coefficient() const
{
  return biot_coefficient;
}  // end get_biot_coefficient



template<int dim>
FluidCouplingStrategy
Model<dim>::coupling_strategy() const
{
  if (solid_model == SolidModelType::Compressibility)
    return FluidCouplingStrategy::None;
  else if (solid_model == SolidModelType::Elasticity)
    return FluidCouplingStrategy::FixedStressSplit;
  else
    AssertThrow(false, ExcNotImplemented());

  return FluidCouplingStrategy::None;
}  // end do_something



template<int dim>
void
Model<dim>::
set_fluid_linear_solver(const LinearSolverType & solver_type)
{
  linear_solver_fluid = solver_type;
}  // end set_fluid_linear_solver( const Model::LinearSolverType solver_type )



template<int dim>
void
Model<dim>::
set_solid_linear_solver(const LinearSolverType & solver_type)
{
  linear_solver_solid = solver_type;
}  // end set_fluid_linear_solver( const Model::LinearSolverType solver_type )

}  // end of namespace

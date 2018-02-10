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

enum FluidModelType {Liquid, SingleGas, /* WaterOil = */ DeadOil,
                     WaterGas, Blackoil};

enum PVTType {Constant, Table, Correlation};

enum Phase {Water, Oil, Gas};


struct ModelConfig
{
  PVTType pvt_oil, pvt_water, pvt_gas;
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
  MPI_Comm                               &mpi_communicator;
  ConditionalOStream                     &pcout;
 public:
  // void read_input(const std::string&,
  //                 const int verbosity_=0);
  // void print_input();

  // Functions of a coordinate
  Function<dim> *get_young_modulus,
    *get_poisson_ratio,
    *get_permeability,
    *get_porosity;
  // *get_porosity;

  // adding data
  void set_fluid_model(FluidModelType type);
  void set_pvt_water(Interpolation::LookupTable &table);
  void set_pvt_oil(Interpolation::LookupTable &table);
  void set_pvt_gas(Interpolation::LookupTable &table);
  void set_rel_perm(const double Sw_crit,
                    const double So_rw,
                    const double k_rw0,
                    const double k_ro0,
                    const double nw,
                    const double no);
  void set_density_sc_w(const double x) {density_sc_w_constant = x;}
  void set_density_sc_o(const double x) {density_sc_o_constant = x;}
  void add_well(const std::string name,
                const double radius,
                const std::vector< Point<dim> > &locations);

  // querying data
  bool has_phase(const Phase &phase) const;
  unsigned int n_phases() const;
  double density_sc_water() const;
  double density_sc_oil() const;
  double gravity() const;
  void get_pvt_oil(const double        pressure,
                   std::vector<double> &dst) const;
  void get_pvt_water(const double        pressure,
                     std::vector<double> &dst) const;
  void get_pvt_gas(const double        pressure,
                   std::vector<double> &dst) const;
  double get_time_step(const double time) const;
  std::vector<int> get_well_ids() const;
  void get_relative_permeability(Vector<double>      &saturation,
                                 std::vector<double> &dst) const;
  int get_well_id(const std::string& well_name) const;

  const Interpolation::LookupTable &
  get_pvt_table_water() const {return pvt_table_water;}
  const Interpolation::LookupTable &
  get_pvt_table_oil() const {return pvt_table_oil;}
  double residual_saturation_water() const;
  double residual_saturation_oil() const;

  // update methods
  void update_well_controls(const double time);
  void locate_wells(const DoFHandler<dim>& dof_handler);
  void update_well_productivities(const Function<dim> &get_pressure,
                                  const Function<dim> &get_saturation);

  void compute_runtime_parameters();
  const std::vector<const Interpolation::LookupTable*> get_pvt_tables() const;

  // ATTRIBUTES
 public:
  const unsigned int                     n_pvt_water_columns = 5;
  const unsigned int                     n_pvt_oil_columns = 5;
  const unsigned int                     n_pvt_gas_columns = 5;
  int                                    initial_refinement_level,
    n_adaptive_steps;
  std::vector<std::pair<double,double>>  local_prerefinement_region;
  Units::Units                           units;
  boost::filesystem::path                mesh_file;
  std::vector< Wellbore<dim> > wells;
  Schedule::Schedule                     schedule;
  double                                 fss_tolerance,
    min_time_step,
    t_max;
  int                                    max_fss_steps;

  FluidModelType                         fluid_model;
  ModelConfig                            config;
 protected:
  std::string                            mesh_file_name,
                                         input_file_name;
  double                                 density_sc_w_constant,
                                         density_sc_o_constant,
                                         porosity,
                                         young_modulus,
                                         poisson_ratio_constant;
  Interpolation::LookupTable             pvt_table_water,
                                         pvt_table_oil,
                                         pvt_table_gas;
  RelativePermeability                   rel_perm;
  std::vector<Phase>                     phases;
 private:
  std::map<double, double>               timestep_table;
  std::map<std::string, int>             well_ids;

  int                                    verbosity;
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
  verbosity = 0;
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
  if (has_phase(Phase::Oil))
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
void Model<dim>::set_fluid_model(FluidModelType model_type)
{
  phases.clear();
  fluid_model = model_type;

  if (fluid_model == FluidModelType::Liquid)
    phases.push_back(Phase::Water);
  else if (fluid_model == DeadOil)
  {
    phases.push_back(Phase::Water);
    phases.push_back(Phase::Oil);
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
double Model<dim>::residual_saturation_water() const
{
  // std::cout << rel_perm.Sw_crit << std::endl;
  return rel_perm.Sw_crit;
}



template <int dim>
inline
double Model<dim>::residual_saturation_oil() const
{
  // std::cout << rel_perm.So_rw;
  return rel_perm.So_rw;
}

}  // end of namespace

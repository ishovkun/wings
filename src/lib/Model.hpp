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
#include <Keywords.h>
#include <LookupTable.hpp>


namespace Model
{
	using namespace dealii;

  enum ModelType {SingleLiquid, SingleGas, WaterOil, WaterGas, Blackoil,
                  SingleLiquidElasticity, SingleGasElasticity,
                  WaterOilElasticity, WaterGasElasticity, BlackoilElasticity};
  enum PVTType {Constant, Table, Correlation};

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

    // Methods getting constant values
    double density_sc_water() const;
    double gravity() const;

    // adding data
    void set_pvt_water(Interpolation::LookupTable &table);
    void set_pvt_oil(Interpolation::LookupTable &table);
    void set_pvt_gas(Interpolation::LookupTable &table);
    void set_density_sc_w(const double x)
    {density_sc_w_constant = x;}
    void add_well(const std::string name,
                  const double radius,
                  const std::vector< Point<dim> > &locations);

    // querying data
    void get_pvt_oil(const double        pressure,
                     std::vector<double> &dst) const;
    void get_pvt_water(const double        pressure,
                       std::vector<double> &dst) const;
    void get_pvt_gas(const double        pressure,
                     std::vector<double> &dst) const;
    double get_time_step(const double time) const;
    std::vector<int> get_well_ids() const;
    int get_well_id(const std::string& well_name) const;
    // update methods
    void update_well_controls(const double time);
    void locate_wells(const DoFHandler<dim>& dof_handler,
                      const FE_DGQ<dim>&     fe);
    void update_well_productivities();
    void compute_runtime_parameters();

    // ATTRIBUTES
  public:
    const unsigned int                     n_pvt_water_columns = 5;
    const unsigned int                     n_pvt_oil_columns = 5;
    const unsigned int                     n_pvt_gas_columns = 5;
    int                                    initial_refinement_level,
                                           n_adaptive_steps;
    std::vector<std::pair<double,double>>  local_prerefinement_region;
    Units::Units                           units;
    Keywords::Keywords                     keywords;
    boost::filesystem::path                mesh_file;
    std::vector< Wellbore::Wellbore<dim> > wells;
    Schedule::Schedule                     schedule;
    double                                 fss_tolerance,
                                           min_time_step,
                                           t_max;
    int                                    max_fss_steps;
    ModelConfig                            config;
  protected:
    std::string                            mesh_file_name, input_file_name;
    double                                 density_sc_w_constant,
                                           porosity,
                                           young_modulus,
                                           poisson_ratio_constant;
    Interpolation::LookupTable             pvt_table_water,
                                           pvt_table_oil,
                                           pvt_table_gas;
  private:
    ParameterHandler                       prm;
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
  void Model<dim>::add_well(const std::string name,
                            const double radius,
                            const std::vector< Point<dim> > &locations)
  {
    Wellbore::Wellbore<dim> w(locations, radius, mpi_communicator);
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


  // template <int dim>
  // void Model<dim>::assign_parameters()
  // {
  //   { // Mesh
  //     prm.enter_subsection(keywords.section_mesh);
  //     mesh_file_name = prm.get(keywords.mesh_file);
  //     mesh_file =
  //       find_file_in_relative_path(mesh_file_name);
  //     // std::cout << "mesh_file "<< mesh_file << std::endl;

  //     initial_refinement_level =
  //       prm.get_integer(keywords.global_refinement_steps);
  //     n_adaptive_steps = prm.get_integer(keywords.adaptive_refinement_steps);

  //     std::vector<double> tmp = Parsers:: parse_string_list<double>
  //       (prm.get(keywords.local_refinement_regions));
  //     local_prerefinement_region.resize(dim);
  //     AssertThrow(tmp.size() == 2*dim,
  //                 ExcMessage("Wrong entry in" +
  //                            keywords.local_refinement_regions));
  //     local_prerefinement_region[0].first = tmp[0];
  //     local_prerefinement_region[0].second = tmp[1];
  //     local_prerefinement_region[1].first = tmp[2];
  //     local_prerefinement_region[1].second = tmp[3];
  //     prm.leave_subsection();
  //   }
  //   { // well data
  //     prm.enter_subsection(keywords.section_wells);
  //     // std::cout << prm.get(keywords.well_parameters) << std::endl;
  //     // assign_wells(prm.get(keywords.well_parameters));
  //     // assign_schedule(prm.get(keywords.well_schedule));
  //     prm.leave_subsection();
  //   }
  //   { // Equation data
  //     prm.enter_subsection(keywords.section_equation_data);
  //     if (prm.get(keywords.unit_system)=="SI")
  //       units.set_system(Units::si_units);
  //     else if (prm.get(keywords.unit_system)=="Field")
  //       units.set_system(Units::field_units);

  //     this->poisson_ratio_constant = prm.get_double(keywords.poisson_ratio);
  //     this->volume_factor_w_constant = prm.get_double(keywords.volume_factor_water);
  //     this->viscosity_w_constant =
  //       prm.get_double(keywords.viscosity_water)*units.viscosity();
  //     this->compressibility_w_constant =
  //       prm.get_double(keywords.compressibility_water)*units.stiffness();
  //     this->density_sc_w_constant =
  //         prm.get_double(keywords.density_sc_water)*units.mass();

  //     // coefficients that are either constant or mapped
  //     Tensor<1,dim> perm_anisotropy = Tensors::get_unit_vector<dim>();
  //     this->get_permeability =
  //       get_hetorogeneous_function_from_parameter(keywords.permeability,
  //                                                 perm_anisotropy);

  //     Tensor<1,dim> stiffness_anisotropy = Tensors::get_unit_vector<dim>();
  //     this->get_young_modulus =
  //       get_hetorogeneous_function_from_parameter(keywords.young_modulus,
  //                                                 stiffness_anisotropy);
  //     Tensor<1,dim> unit_vector = Tensors::get_unit_vector<dim>();
  //     this->get_porosity =
  //       get_hetorogeneous_function_from_parameter(keywords.porosity,
  //                                                 unit_vector);
  //     prm.leave_subsection();
  //   }
  //   { // Solver
  //     prm.enter_subsection(keywords.section_solver);
  //     this->t_max = prm.get_double(keywords.t_max);
  //     this->min_time_step = prm.get_double(keywords.minimum_time_step);
  //     this->fss_tolerance = prm.get_double(keywords.fss_tolerance);
  //     this->max_fss_steps = prm.get_integer(keywords.max_fss_steps);
  //     this->parse_time_stepping();
  //     prm.leave_subsection();
  //   }
  // }  // eom


  template <int dim>
  void Model<dim>::update_well_controls(const double time)
  {
    for (unsigned int i=0; i<wells.size(); i++)
      wells[i].set_control(schedule.get_control(time, i));
  } // eom


  template <int dim>
  void Model<dim>::locate_wells(const DoFHandler<dim>& dof_handler,
                                   const FE_DGQ<dim>&     fe)
  {
    for (unsigned int i=0; i<wells.size(); i++)
    {
      // std::cout << "well " << i << std::endl;
      wells[i].locate(dof_handler, fe);
    }
  } // eom


  template <int dim>
  void Model<dim>::update_well_productivities()
  {
    for (auto & well : wells)
      well.update_productivity(this->get_permeability);
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
    AssertThrow(dst.size() == 5, ExcDimensionMismatch(dst.size(), 5));
    pvt_table_water.get_values(pressure, dst);
  }  // eom


  template <int dim>
  void Model<dim>::get_pvt_oil(const double        pressure,
                               std::vector<double> &dst) const
  {
    AssertThrow(dst.size() == 5, ExcDimensionMismatch(dst.size(), 5));
    pvt_table_oil.get_values(pressure, dst);
  }  // eom
}  // end of namespace

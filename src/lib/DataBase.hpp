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
#include <Units.cc>
#include <Tensors.hpp>
#include <Keywords.cc>


namespace Data
{
	using namespace dealii;

  template <int dim>
  class DataBase
  {
  public:
    DataBase();
    // ~DataBase();
    void read_input(const std::string&);

    // Functions of a coordinate
    Function<dim> *get_young_modulus,
                  *get_poisson_ratio,
                  *get_permeability,
                  *get_porosity;
    // Methods getting constant values
    double viscosity_water() const;
    double volume_factor_water() const;
    double compressibility_water() const;
    // Methods getting pressure-dependent values
    // double get_viscosity(const double pressure) const;
    // double get_volume_factor(const double pressure) const;
    // double get_compressibility(const double pressure) const;

    double get_time_step(const double time) const;
    std::vector<int> get_well_ids() const;
    void update_well_controls(const double time);
    void locate_wells(const DoFHandler<dim>& dof_handler,
                      const FE_DGQ<dim>&     fe);
    void update_well_transmissibilities();


  private:
    void declare_parameters();
    void assign_parameters();
    void compute_runtime_parameters();
    void check_input();
    Function<dim>*
    get_hetorogeneous_function_from_parameter(const std::string&   par_name,
                                              const Tensor<1,dim>& anisotropy);
    boost::filesystem::path find_file_in_relative_path(const std::string fname);
    void parse_time_stepping();
    void assign_wells(const std::string& text);
    void assign_schedule(const std::string& text);
    int get_well_id(const std::string& well_name) const;

    // ATTRIBUTES
  public:
    int                                    initial_refinement_level,
                                           n_prerefinement_steps,
                                           n_adaptive_steps;
    std::vector<std::pair<double,double>>  local_prerefinement_region;
    Units::Units                           units;
    Keywords::Keywords                     keywords;
    boost::filesystem::path                mesh_file;
    std::vector< Wellbore::Wellbore<dim> > wells;
    Schedule::Schedule                     schedule;
  private:
    std::string                            mesh_file_name, input_file_name;
    double                                 volume_factor_w_constant,
                                           viscosity_w_constant,
                                           compressibility_w_constant,
                                           porosity,
                                           young_modulus,
                                           poisson_ratio_constant;
    double                                 fss_tolerance,
                                           min_time_step,
                                           t_max;
    int                                    max_fss_steps;
    ParameterHandler                       prm;
    std::map<double, double>               timestep_table;
    std::map<std::string, int>             well_ids;
  };  // eom


  template <int dim>
  DataBase<dim>::DataBase()
  {
    declare_parameters();
  }  // eom


  template <int dim>
  void DataBase<dim>::read_input(const std::string& file_name)
  {
    std::cout << "Reading " << file_name << std::endl;
    input_file_name = file_name;
    prm.parse_input(file_name);
    prm.print_parameters(std::cout, ParameterHandler::Text);
    assign_parameters();
    // compute_runtime_parameters();
    // check_input();
  }  // eom


  template <int dim>
  double DataBase<dim>::compressibility_water() const
  {
    return this->compressibility_w_constant;
  }  // eom


  template <int dim>
  double DataBase<dim>::viscosity_water() const
  {
    return this->viscosity_w_constant;
  }  // eom


  template <int dim>
  double DataBase<dim>::volume_factor_water() const
  {
    return this->volume_factor_w_constant;
  }  // eom


  template <int dim>
  void DataBase<dim>::parse_time_stepping()
  {
    // Parse time stepping table
    std::vector<Point<2> > tmp =
      Parsers::parse_point_list<2>(prm.get(keywords.time_stepping));
    for (const auto &row : tmp)
      this->timestep_table[row[0]] = row[1];
  } // eom


  template <int dim>
  double DataBase<dim>::get_time_step(const double time) const
  /* get value of the time step from the time-stepping table */
  {
    double time_step = timestep_table.rbegin()->second;
    for (const auto &it : timestep_table)
      {
        if (time >= it.first)
          time_step = it.second;
        else
          break;
      }

    return time_step;
  }  // eom


  template <int dim>
  void DataBase<dim>::declare_parameters()
  {
    { // Mesh
      prm.enter_subsection(keywords.section_mesh);
      prm.declare_entry(keywords.mesh_file,
                        "", Patterns::Anything());
      prm.declare_entry(keywords.global_refinement_steps,
                        "0", Patterns::Integer(0, 100));
      prm.declare_entry(keywords.adaptive_refinement_steps,
                        "0", Patterns::Integer(0, 100));
      prm.declare_entry(keywords.local_refinement_regions,
                        "", Patterns::List(Patterns::Double()));
      prm.leave_subsection();
    }

    { // wells
      prm.enter_subsection(keywords.section_wells);
      prm.declare_entry(keywords.well_parameters,
                        "", Patterns::Anything());
      prm.declare_entry(keywords.well_schedule,
                        "", Patterns::Anything());
      prm.leave_subsection();
    }

    { // equation data
      prm.enter_subsection(keywords.section_equation_data);
      // Constant parameters
      prm.declare_entry(keywords.unit_system, "SI",
                        Patterns::Selection("SI|Field"));
      prm.declare_entry(keywords.young_modulus, "1e9",
                        Patterns::Anything());
      prm.declare_entry(keywords.poisson_ratio, "0.3",
                        Patterns::Double(0, 0.5));
      prm.declare_entry(keywords.volume_factor_water, "1",
                        Patterns::Anything());
      prm.declare_entry(keywords.viscosity_water, "1e-3",
                        Patterns::Double());
      prm.declare_entry(keywords.compressibility_water, "1e-8",
                        Patterns::Double());
      prm.declare_entry(keywords.permeability, "1e-12",
                        Patterns::Anything());
      prm.declare_entry(keywords.porosity, "0.3",
                        Patterns::Anything());
      prm.leave_subsection();
    }
    { // Solver
      prm.enter_subsection(keywords.section_solver);
      prm.declare_entry(keywords.t_max, "1",
                        Patterns::Double());
      prm.declare_entry(keywords.time_stepping, "(0, 1e-3)",
                        Patterns::Anything());
      prm.declare_entry(keywords.minimum_time_step, "1e-9",
                        Patterns::Double());
      prm.declare_entry(keywords.fss_tolerance, "1e-9",
                        Patterns::Double());
      prm.declare_entry(keywords.max_fss_steps, "30",
                        Patterns::Integer());
      // prm.declare_entry("Newton tolerance", "1e-9", Patterns::Double());
      // prm.declare_entry("Max Newton steps", "20", Patterns::Integer());
      prm.leave_subsection();
    }
  }  // eom


  template <int dim>
  Function<dim> *
  DataBase<dim>::
  get_hetorogeneous_function_from_parameter(const std::string&   par_name,
                                            const Tensor<1,dim>& anisotropy)
  {
    const std::string entry = prm.get(par_name);
    if (Parsers::is_number(entry))
      {
        std::vector<double> quantity;
        for (int c=0; c<dim; c++)
          quantity.push_back(boost::lexical_cast<double>(entry)*anisotropy[c]);
        return new ConstantFunction<dim>(quantity);
      }
    else
    {
      std::cout << "Searching " << par_name << std::endl;
      boost::filesystem::path input_file_path(input_file_name);
      boost::filesystem::path data_file =
        input_file_path.parent_path() / entry;
      return new BitMap::BitMapFunction<dim>(data_file.string(),
                                             anisotropy);
    }
  }  // eom


  template <int dim>
  boost::filesystem::path
  DataBase<dim>::find_file_in_relative_path(const std::string fname)
  {
    boost::filesystem::path input_file_path(input_file_name);
    std::cout << "Searching " << fname << std::endl;
    boost::filesystem::path data_file =
      input_file_path.parent_path() / fname;
    std::cout << "Found " << data_file << std::endl;
    return data_file;
  }  // eom


  template <int dim>
  int DataBase<dim>::get_well_id(const std::string& well_name) const
  {
    return well_ids.find(well_name)->second;
  } // eom


  template <int dim>
  std::vector<int> DataBase<dim>::get_well_ids() const
  {
    std::vector<int> result;
    for(auto & id : well_ids)
      result.push_back(id.second);
    return result;
  } // eom


  template <int dim>
  void DataBase<dim>::assign_schedule(const std::string& text)
  {
    /*
      Split first based on ";" - schedule entries
      Then split based on "," - parameters of schedule as follows:
      time, well_name, control type id, control value, others (skin etc.)
    */

    std::vector<std::string> lines;
    boost::algorithm::split(lines, text, boost::is_any_of(";"));
    for (auto & line : lines)
    {
      // std::cout << line << std::endl;
      std::vector<std::string> entries;
      boost::algorithm::split(entries, line, boost::is_any_of(","));

      // Handle case when the last entry in schedule ends with ";"
      // Boost thinks that there is something after
      if (entries.size() == 1 && entries[0].size() == 0)
        break;

      // Process entries
      AssertThrow(entries.size() >= 4,
                  ExcMessage("Wrong entry in schedule"));

      Schedule::ScheduleEntry schedule_entry;

      // get time
      schedule_entry.time = Parsers::convert<double>(entries[0]);

      // get well name and identifier
      std::string well_name = entries[1];
      // std::cout << "well name = " << well_name << std::endl;
      boost::algorithm::trim(well_name);
      // schedule_entry.well_name = well_name;
      schedule_entry.well_id = get_well_id(well_name);

      // get control type
      const int control_type_id = Parsers::convert<int>(entries[2]);
      schedule_entry.control.type =
        Schedule::well_control_type_indexing.find(control_type_id)->second;

      // get control value
      schedule_entry.control.value = Parsers::convert<double>(entries[3]);

      // get skin
      if (entries.size() > 4)
      {
        schedule_entry.control.skin = Parsers::convert<double>(entries[4]);
      }

      schedule.add_entry(schedule_entry);
    }
  }  // eom


  template <int dim>
  void DataBase<dim>::assign_wells(const std::string& text)
  {
    /*
      Keyword structure is as follows:
      string well_name
      double radius
      comma-semicolon-separated list locations
      example
      [Well1, 0.1, 1, (1,1; 2,2)], [Well2, 0.1, 1, (1,1; 2,2)]
    */
    const auto & delim = std::pair<std::string,std::string>("[","]");
    std::vector<std::string> split =
      Parsers::split_bracket_group(text, delim);

    for (auto & item : split)
    {
      // std::cout << item << std::endl;
      const auto & str_params = Parsers::split_ignore_brackets(item);
      const std::string well_name = str_params[0];
      const double radius = Parsers::convert<double>(str_params[1]);
      // separate separate points
      std::vector<std::string> point_strs;
      std::vector< Point<dim> > locations;
      boost::algorithm::split(point_strs, str_params[2], boost::is_any_of(";"));
      for (auto & point_str : point_strs)
      {
        const auto & point = Parsers::parse_string_list<double>(point_str);
        AssertThrow(point.size() == dim,
                    ExcMessage("Dimensions don't match"));
        Point<dim> p;
        for (int d=0; d<dim; d++)
          p[d] = point[d];
        locations.push_back(p);
      }

      Wellbore::Wellbore<dim> w(locations, radius);
      this->wells.push_back(w);

      // check if well_id is in unique_well_ids and add if not
      if (well_ids.empty())
        well_ids[well_name] = 0;
      else
      {
        std::map<std::string, int>::iterator
          it = well_ids.begin(),
          end = well_ids.end();

        for (; it!=end; it++)
        {
          AssertThrow(it->first != well_name, ExcMessage("Duplicates in wells"));
        }

        const int id = well_ids.size();
        well_ids[well_name] = id;
      }
    }
  }  // eom


  template <int dim>
  void DataBase<dim>::assign_parameters()
  {
    { // Mesh
      prm.enter_subsection(keywords.section_mesh);
      mesh_file_name = prm.get(keywords.mesh_file);
      mesh_file =
        find_file_in_relative_path(mesh_file_name);
      // std::cout << "mesh_file "<< mesh_file << std::endl;

      initial_refinement_level =
        prm.get_integer(keywords.global_refinement_steps);
      n_adaptive_steps = prm.get_integer(keywords.adaptive_refinement_steps);

      std::vector<double> tmp = Parsers:: parse_string_list<double>
        (prm.get(keywords.local_refinement_regions));
      local_prerefinement_region.resize(dim);
      AssertThrow(tmp.size() == 2*dim,
                  ExcMessage("Wrong entry in" +
                             keywords.local_refinement_regions));
      local_prerefinement_region[0].first = tmp[0];
      local_prerefinement_region[0].second = tmp[1];
      local_prerefinement_region[1].first = tmp[2];
      local_prerefinement_region[1].second = tmp[3];
      prm.leave_subsection();
    }
    { // well data
      prm.enter_subsection(keywords.section_wells);
      assign_wells(prm.get(keywords.well_parameters));
      assign_schedule(prm.get(keywords.well_schedule));
      prm.leave_subsection();
    }
    { // Equation data
      prm.enter_subsection(keywords.section_equation_data);
      if (prm.get(keywords.unit_system)=="SI")
        units.set_system(Units::si_units);
      else if (prm.get(keywords.unit_system)=="Field")
        units.set_system(Units::field_units);

      this->poisson_ratio_constant = prm.get_double(keywords.poisson_ratio);
      this->volume_factor_w_constant = prm.get_double(keywords.volume_factor_water);
      this->viscosity_w_constant =
        prm.get_double(keywords.viscosity_water)*units.viscosity();
      this->compressibility_w_constant =
        prm.get_double(keywords.compressibility_water)*units.stiffness();

      // coefficients that are either constant or mapped
      Tensor<1,dim> perm_anisotropy = Tensors::get_unit_vector<dim>();
      this->get_permeability =
        get_hetorogeneous_function_from_parameter(keywords.permeability,
                                                  perm_anisotropy);

      Tensor<1,dim> stiffness_anisotropy = Tensors::get_unit_vector<dim>();
      this->get_young_modulus =
        get_hetorogeneous_function_from_parameter(keywords.young_modulus,
                                                  stiffness_anisotropy);
      Tensor<1,dim> unit_vector = Tensors::get_unit_vector<dim>();
      this->get_porosity =
        get_hetorogeneous_function_from_parameter(keywords.porosity,
                                                  unit_vector);
      prm.leave_subsection();
    }
    { // Solver
      prm.enter_subsection(keywords.section_solver);
      this->t_max = prm.get_double(keywords.t_max);
      this->min_time_step = prm.get_double(keywords.minimum_time_step);
      this->fss_tolerance = prm.get_double(keywords.fss_tolerance);
      this->max_fss_steps = prm.get_integer(keywords.max_fss_steps);
      this->parse_time_stepping();
      prm.leave_subsection();
    }
  }  // eom


  template <int dim>
  void DataBase<dim>::update_well_controls(const double time)
  {
    for (unsigned int i=0; i<wells.size(); i++)
      wells[i].set_control(schedule.get_control(time, i));
  } // eom


  template <int dim>
  void DataBase<dim>::locate_wells(const DoFHandler<dim>& dof_handler,
                                   const FE_DGQ<dim>&     fe)
  {
    for (unsigned int i=0; i<wells.size(); i++)
      wells[i].locate(dof_handler, fe);
  } // eom

  template <int dim>
  void DataBase<dim>::update_well_transmissibilities()
  {
    for (auto & well : wells)
      well.update_transmissibility(this->get_permeability);
  }  // eom
}  // end of namespace

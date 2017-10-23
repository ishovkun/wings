#pragma once

// #include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <boost/filesystem.hpp>

// Custom modules
#include <Parsers.hpp>
#include <BitMap.hpp>
#include <Units.cc>
#include <Tensors.hpp>
#include <Keywords.cpp>


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
    double get_viscosity() const;
    double get_volume_factor() const;
    double get_compressibility() const;
    // Methods getting pressure-dependent values
    double get_viscosity(const double pressure) const;
    double get_volume_factor(const double pressure) const;
    double get_compressibility(const double pressure) const;
  private:
    void declare_parameters();
    void assign_parameters();
    void compute_runtime_parameters();
    void check_input();
    Function<dim>*
    get_hetorogeneous_function_from_parameter(const std::string&   par_name,
                                              const Tensor<1,dim>& anisotropy);

    // ATTRIBUTES
  public:
    int                                   initial_refinement_level,
                                          n_prerefinement_steps,
                                          n_adaptive_steps;
    std::vector<std::pair<double,double>> local_prerefinement_region;
    Units::Units                          units;
    Keywords::Keywords                    keywords;
  private:
    std::string                           mesh_file_name, input_file_name;
    double                                volume_factor_w,
                                          viscosity_w,
                                          porosity,
                                          compressibility_w,
                                          young_modulus,
                                          poisson_ratio;
    double                                fss_tolerance,
                                          min_time_step,
                                          t_max;
    int                                   max_fss_steps;
    ParameterHandler                      prm;
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
  double DataBase<dim>::get_compressibility() const
  {
    return this->fluid_compressibility;
  }  // eom


  template <int dim>
  double DataBase<dim>::get_viscosity() const
  {
    return this->viscosity;
  }  // eom


  template <int dim>
  double DataBase<dim>::get_volume_factor() const
  {
    return this->volume_factor;
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

    { // equation data
      prm.enter_subsection(keywords.section_equation_data);
      // Constant parameters
      prm.declare_entry(keywords.young_modulus,
                        "1e9", Patterns::Anything());
      prm.declare_entry(keywords.poisson_ratio,
                        "0.3", Patterns::Double(0, 0.5));
      prm.declare_entry(keywords.volume_factor_water,
                        "1", Patterns::Anything());
      prm.declare_entry(keywords.viscosity_water,
                        "1e-3", Patterns::Double());
      prm.declare_entry(keywords.compressibility_water,
                        "1e-8", Patterns::Double());
      prm.declare_entry(keywords.permeability,
                        "1e-12", Patterns::Anything());
      prm.leave_subsection();
    }
    { // Solver
      prm.enter_subsection(keywords.section_solver);
      prm.declare_entry(keywords.t_max,
                        "1", Patterns::Double());
      prm.declare_entry(keywords.time_stepping,
                        "(0, 1e-3)", Patterns::Anything());
      prm.declare_entry(keywords.minimum_time_step,
                        "1e-9", Patterns::Double());
      prm.declare_entry(keywords.fss_tolerance,
                        "1e-9", Patterns::Double());
      prm.declare_entry(keywords.max_fss_steps,
                        "30", Patterns::Integer());
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
      std::cout << "Reading " << data_file << std::endl;
      return new BitMap::BitMapFunction<dim>(data_file.string(),
                                             anisotropy);
    }
  }  // eom

  template <int dim>
  void DataBase<dim>::assign_parameters()
  {
    { // Mesh
      prm.enter_subsection(keywords.section_mesh);
      mesh_file_name = prm.get(keywords.mesh_file);
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
    { // Equation data
      prm.enter_subsection(keywords.section_equation_data);

      this->poisson_ratio = prm.get_double(keywords.poisson_ratio);
      this->volume_factor_w = prm.get_double(keywords.volume_factor_water);
      this->viscosity_w = prm.get_double(keywords.viscosity_water);
      this->compressibility_w = prm.get_double(keywords.compressibility_water);

      // coefficients that are either constant or mapped
      Tensor<1,dim> perm_anisotropy = Tensors::get_unit_vector<dim>();
      this->get_permeability =
        get_hetorogeneous_function_from_parameter(keywords.permeability,
                                                  perm_anisotropy);

      Tensor<1,dim> stiffness_anisotropy = Tensors::get_unit_vector<dim>();
      this->get_young_modulus =
        get_hetorogeneous_function_from_parameter(keywords.young_modulus,
                                                  stiffness_anisotropy);

      // test output
      // std::cout
      //   << this->get_permeability->value(Point<dim>(1,1), 1)
      //   << std::endl;
      prm.leave_subsection();
    }
    { // Solver
      prm.enter_subsection(keywords.section_solver);
      this->t_max = prm.get_double(keywords.t_max);
      this->min_time_step = prm.get_double(keywords.minimum_time_step);
      this->fss_tolerance = prm.get_double(keywords.fss_tolerance);
      this->max_fss_steps = prm.get_integer(keywords.max_fss_steps);
      prm.leave_subsection();
    }
  }  // eom
}  // end of namespace

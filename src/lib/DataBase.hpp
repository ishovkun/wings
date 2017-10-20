# pragma once

// #include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

// Custom modules
#include <Parsers.hpp>


namespace Data
{
	using namespace dealii;

  template <int dim>
  class DataBase
  {
  public:
    DataBase();
    // ~DataBase();
    void read_input(std::string);

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

    // ATTRIBUTES
  public:
    int initial_refinement_level, n_prerefinement_steps, n_adaptive_steps;
    std::vector<std::pair<double,double>> local_prerefinement_region;
  private:
    double volume_factor, viscosity, porosity, fluid_compressibility,
           young_modulus, poisson_ratio;
    std::vector<double> permeability;
    ParameterHandler    prm;


  };  // eom


  template <int dim>
  DataBase<dim>::DataBase()
  {
    for (int d=0; d<dim; d++)
      this->permeability.push_back(1);

    this->porosity = 0.3;
    this->volume_factor = 1;
    this->viscosity = 1e-3;
    this->young_modulus = 1;
    this->poisson_ratio = 0.3;
    this->fluid_compressibility = 1e-8;

    this->get_young_modulus = new ConstantFunction<dim>(young_modulus);
    this->get_poisson_ratio = new ConstantFunction<dim>(poisson_ratio);
    this->get_porosity = new ConstantFunction<dim>(porosity);
    this->get_permeability = new ConstantFunction<dim>(permeability);

    declare_parameters();
  }  // eom


  template <int dim>
  void DataBase<dim>::read_input(std::string file_name)
  {
    std::cout << "Reading " << file_name << std::endl;
    prm.read_input(file_name);
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
      prm.enter_subsection("Mesh");
      prm.declare_entry("Mesh file", "", Patterns::Anything());
      prm.declare_entry("Initial global refinement steps", "0", Patterns::Integer(0, 100));
      prm.declare_entry("Adaptive steps", "0", Patterns::Integer(0, 100));
      prm.declare_entry("Local refinement region", "",
                        Patterns::List(Patterns::Double()));
      prm.leave_subsection();
    }

    { // equation data
      prm.enter_subsection("Equation data");
      // Constant parameters
      prm.declare_entry("Young modulus", "1e9", Patterns::Anything());
      prm.declare_entry("Poisson ratio", "0.3", Patterns::Double(0, 0.5));
      prm.declare_entry("Volume factor water", "1", Patterns::Anything());
      prm.declare_entry("Viscosity water", "1e-3", Patterns::Double());
      prm.declare_entry("Compressibility water", "1e-8", Patterns::Double());
      prm.declare_entry("Permeability", "1e-12", Patterns::Anything());
      prm.leave_subsection();
    }
    { // Solver
      prm.enter_subsection("Solver");
      prm.declare_entry("T max", "1", Patterns::Double());
      prm.declare_entry("Time stepping", "(0, 1e-3)", Patterns::Anything());
      prm.declare_entry("Minimum time step", "1e-9", Patterns::Double());
      prm.declare_entry("FSS tolerance", "1e-9", Patterns::Double());
      prm.declare_entry("Max FSS steps", "30", Patterns::Integer());
      // prm.declare_entry("Newton tolerance", "1e-9", Patterns::Double());
      // prm.declare_entry("Max Newton steps", "20", Patterns::Integer());
      prm.leave_subsection();
    }
  }  // eom


  template <int dim>
  void DataBase<dim>::assign_parameters()
  {
    { // Mesh
      prm.enter_subsection("Mesh");
      mesh_file_name = prm.get("Mesh file");
      initial_refinement_level = prm.get_integer("Initial global refinement steps");
      n_adaptive_steps = prm.get_integer("Adaptive steps");
      std::vector<double> tmp =
        Parsers::parse_string_list<double>(prm.get("Local refinement region"));
      local_prerefinement_region.resize(dim);
      AssertThrow(tmp.size() == 2*dim,
                  ExcMessage("Wrong entry in Local refinement region"));
      local_prerefinement_region[0].first = tmp[0];
      local_prerefinement_region[0].second = tmp[1];
      local_prerefinement_region[1].first = tmp[2];
      local_prerefinement_region[1].second = tmp[3];
      prm.leave_subsection();
    }
  }  // eom
}  // end of namespace

# pragma once

// #include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <boost/filesystem.hpp>

// Custom modules
#include <Parsers.hpp>
#include <BitMap.hpp>


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
    void assign_permeability_param();

    // ATTRIBUTES
  public:
    int initial_refinement_level, n_prerefinement_steps, n_adaptive_steps;
    std::vector<std::pair<double,double>> local_prerefinement_region;
  private:
    std::string mesh_file_name, input_file_name;
    double volume_factor_w, viscosity_w, porosity, compressibility_w,
           young_modulus, poisson_ratio;
    // std::vector<double> permeability;
    ParameterHandler    prm;


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
  void DataBase<dim>::assign_permeability_param()
  {
    const std::string perm_entry = prm.get("Permeability");
    if (Parsers::is_number(perm_entry))
      {
        std::vector<double> perm_tensor;
        for (int d=0; d<dim; d++)
          perm_tensor.push_back(boost::lexical_cast<double>(perm_entry));
        this->get_permeability = new ConstantFunction<dim>(perm_tensor);
      }
    else
      {
        std::cout << "Searching " << perm_entry << std::endl;
        boost::filesystem::path input_file_path(input_file_name);
        boost::filesystem::path perm_file =
          input_file_path.parent_path() / perm_entry;
        std::cout << "Reading " << perm_file << std::endl;
        this->get_permeability =
          new BitMap::BitMapFunction<dim>(perm_file.string());
        // just test what it gives
        // Point<dim> p(1, 1);
        // std::cout << this->get_permeability->value(p, 0) << std::endl;
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
    {  // Equation data
      prm.enter_subsection("Equation data");

      this->poisson_ratio = prm.get_double("Poisson ratio");
      this->young_modulus = prm.get_double("Young modulus");
      this->volume_factor_w = prm.get_double("Volume factor water");
      this->viscosity_w = prm.get_double("Viscosity water");
      this->compressibility_w = prm.get_double("Compressibility water");

      // coefficients that are either constant or mapped
      assign_permeability_param();
      Point<dim> p(1, 1);
      std::cout << this->get_permeability->value(p, 1) << std::endl;
      // const std::string perm_entry = prm.get("Permeability");
      // if (Parsers::is_number(perm_entry))
      // {
      //   std::vector<double> perm_tensor;
      //   for (int d=0; d<dim; d++)
      //     perm_tensor.push_back(boost::lexical_cast<double>(perm_entry));
      //   this->get_permeability = new ConstantFunction<dim>(perm_tensor);
      // }
      // else
      // {
      //   std::cout << "Searching " << perm_entry << std::endl;
      //   boost::filesystem::path input_file_path(input_file_name);
      //   boost::filesystem::path perm_file =
      //     input_file_path.parent_path() / perm_entry;
      //   std::cout << "Reading " << perm_file << std::endl;
      //   this->get_permeability =
      //     new BitMap::BitMapFunction<dim>(perm_file.string());
      //   // just test what it gives
      //   // Point<dim> p(1, 1);
      //   // std::cout << this->get_permeability->value(p, 0) << std::endl;
      // }

      // if (this->uniform_young_modulus)
      //   this->young_modulus = prm.get_double("Young modulus");
      // else
      //   {
      //     std::vector<double> tmp;
      //     tmp.resize(2);
      //     tmp = Parsers::parse_string_list<double>(prm.get("Young modulus range"));
      //     this->young_modulus_limits.first = tmp[0];
      //     this->young_modulus_limits.second = tmp[1];
      //   }

      // this->regularization_parameter_kappa = prm.get_double("Regularization kappa");
      // this->penalty_parameter = prm.get_double("Penalization c");
      // std::vector<double> tmp =
      //   Parsers::parse_string_list<double>(prm.get("Regularization epsilon"));
      // regularization_epsilon_coefficients.first = tmp[0];
      // regularization_epsilon_coefficients.second = tmp[1];
      // // Ranges
      // // tmp.clear();
      // // Bitmap file
      // bitmap_file_name = prm.get("Bitmap file");
      // std::vector<double> tmp1 =
      //   Parsers::parse_string_list<double>(prm.get("Bitmap range"));
      // bitmap_range.resize(dim);
      // // std::cout << tmp1.size() << std::endl;
      // if (tmp1.size() > 0)
      //   for (int i=0; i<dim; ++i)
      //     {
      //       bitmap_range[i].first = tmp1[2*i];
      //       bitmap_range[i].second = tmp1[2*i + 1];
      //     }
      // prm.leave_subsection();
    }
  }  // eom
}  // end of namespace

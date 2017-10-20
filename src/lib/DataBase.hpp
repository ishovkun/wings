# pragma once

// #include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>

namespace Data
{
	using namespace dealii;

  template <int dim>
  class DataBase
  {
  public:
    DataBase();
    // ~DataBase();

    double volume_factor, viscosity, porosity, fluid_compressibility,
      young_modulus, poisson_ratio;
    std::vector<double> permeability;

    Function<dim> *get_young_modulus,
                  *get_poisson_ratio,
                  *get_permeability,
                  *get_porosity;

    double get_viscosity() const;
    double get_volume_factor() const;
    double get_compressibility() const;

    double get_viscosity(const double pressure) const;
    double get_volume_factor(const double pressure) const;
    double get_compressibility(const double pressure) const;

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


}  // end of namespace

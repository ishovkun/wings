# pragma once

// #include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

namespace Data
{
	using namespace dealii;

  template <int dim>
  class DataBase
  {
  public:
    // DataBase();
    // ~DataBase();

    // Permeability methods
    void set_permeability(const Tensor<1,dim> &k);
    void set_permeability(const std::vector< Vector<double> > &k);
    void set_permeability_x(const Vector<double> &k);
    void set_permeability_y(const Vector<double> &k);
    void set_permeability_z(const Vector<double> &k);
    // permeability(typename DoFHandler<dim>::active_cell_iterator &cell)
    void permeability(const unsigned int cell_index,
                      Tensor<1,dim>      &out) const;

    // Viscosity methods
    void set_viscosity(const double &mu);
    double viscosity(const unsigned int cell_index) const;

    // Formation fluid volume factor B_f
    void set_volume_factor(const double volume_factor);
    double volume_factor(const unsigned int cell_index) const;

    // Porosity
    void set_porosity(const double phi);
    double porosity(const unsigned int cell_index) const;

    // Fluid compressibility
    void set_fluid_compressibility(const double cf);
    double fluid_compressibility(const unsigned int cell_index) const;
  };  // eom


  template <int dim>
  void DataBase<dim>::permeability(const unsigned int cell_index,
                                   Tensor<1,dim>      &out) const
  {
    out = 0;
    for (int d=0; d<dim; d++)
      out[d] = 1;
  } // eom


  template <int dim>
  double DataBase<dim>::viscosity(const unsigned int cell_index) const
  {
    return 1.0;
  } // eom


  template <int dim>
  double DataBase<dim>::volume_factor(const unsigned int cell_index) const
  {
    return 1.0;
  } // eom


  template <int dim>
  double DataBase<dim>::porosity(const unsigned int cell_index) const
  {
    return 1.0;
  } // eom


  template <int dim>
  double DataBase<dim>::fluid_compressibility(const unsigned int cell_index) const
  {
    return 1.0;
  } // eom

}  // end of namespace

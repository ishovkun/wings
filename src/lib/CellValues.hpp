#pragma once

#include <DataBase.hpp>
#include <Math.hpp>
// #include <DefaultValues.cc>

namespace CellValues
{
  using namespace dealii;

  template <int dim>
  using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;


  template <int dim>
  class CellValuesBase
  {
  public:
    CellValuesBase(const Data::DataBase<dim> &data_);
    virtual void update(const CellIterator<dim> &cell);
    virtual double get_mass_matrix_entry() const;
    virtual void update_face_values(const CellValuesBase<dim> &neighbor_data,
                                    const Tensor<1,dim>       &dx,
                                    const Tensor<1,dim>       &face_normal,
                                    const double              dS);


   public:
    double Q, J, T_face, G_face;
   private:
    const Data::DataBase<dim>  &data;
    double phi, mu_w, B_w, C_w, cell_volume;
    Vector<double> k;
  };


  template <int dim>
  CellValuesBase<dim>::CellValuesBase(const Data::DataBase<dim> &data_)
    :
    data(data_),
    k(dim)
  {}


  template <int dim>
  void
  CellValuesBase<dim>::update(const CellIterator<dim> &cell)
  {
    data.get_permeability->vector_value(cell->center(), k);
    mu_w = data.viscosity_water();
    B_w = data.volume_factor_water();
    phi = data.get_porosity->value(cell->center());
    C_w = data.compressibility_water();
    cell_volume = cell->measure();
    // calculate source term
    Q = 0;
    J = 0;
    for (const auto & well : data.wells)
    {
      std::pair<double,double> J_and_Q = well.get_J_and_Q(cell);
      J += J_and_Q.first;
      Q += J_and_Q.second;
    }
  } // eom


  template <int dim>
  double
  CellValuesBase<dim>::
  get_mass_matrix_entry() const
  {
    return cell_volume/B_w*(phi*C_w);
  }


  template <int dim>
  void CellValuesBase<dim>::
  // face_transmissibility(const CellValuesBase<dim> &neighbor_data,
  update_face_values(const CellValuesBase<dim> &neighbor_data,
                     const Tensor<1,dim>       &dx,
                     const Tensor<1,dim>       &face_normal,
                     const double              face_area)
  {
    T_face = 0;
    G_face = 0;

    Vector<double> k_face(3);
    Math::harmonic_mean(k, neighbor_data.k, k_face);
    const double mu_w_face = Math::arithmetic_mean(mu_w, neighbor_data.mu_w);
    const double B_w_face = Math::arithmetic_mean(B_w, neighbor_data.B_w);

    double distance = dx.norm(); // to normalize
    if (distance == 0.0)
      return;

    for (int d=0; d<dim; ++d)
      {
        if (abs(dx[d]/distance) > DefaultValues::small_number)
          {
            T_face += 1./mu_w_face/B_w_face *
                (k_face[d]*face_normal[d]/dx[d])*face_area;
          }
      }

    G_face = data.density_sc_water()/B_w_face*data.gravity()*T_face*dx[2]*face_normal[2];

    // return T;

  } // eom

}  // end of namespace

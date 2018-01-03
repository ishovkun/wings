#pragma once

#include <Model.hpp>
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
    CellValuesBase(const Model::Model<dim> &model_);
    virtual void update(const CellIterator<dim> &cell);
    virtual double get_mass_matrix_entry() const;
    virtual void update_face_values(const CellValuesBase<dim> &neighbor_data,
                                    const Tensor<1,dim>       &dx,
                                    const Tensor<1,dim>       &face_normal,
                                    const double              dS);


   public:
    double Q, J, T_face, G_face;
   private:
    const Model::Model<dim>  &model;
    double phi, mu_w, B_w, C_w, cell_volume;
    Vector<double> k;
    std::vector<double> pvt_values;
  };


  template <int dim>
  CellValuesBase<dim>::CellValuesBase(const Model::Model<dim> &model_)
    :
    model(model_),
    k(dim),
    pvt_values(model.n_pvt_water_columns)
  {}


  template <int dim>
  void
  CellValuesBase<dim>::update(const CellIterator<dim> &cell)
  {
    model.get_permeability->vector_value(cell->center(), k);
    model.get_pvt_water(0.0, pvt_values);
    B_w = pvt_values[0];
    C_w = pvt_values[1];
    mu_w = pvt_values[2];
    // mu_w = model.viscosity_water();
    // B_w = model.volume_factor_water();
    phi = model.get_porosity->value(cell->center());
    // C_w = model.compressibility_water();
    cell_volume = cell->measure();
    // calculate source term
    Q = 0;
    J = 0;
    for (const auto & well : model.wells)
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

    // G_face = data.density_sc_water()/B_w_face*data.gravity()*T_face*dx[2]*face_normal[2];
    G_face = model.density_sc_water()/B_w_face/B_w_face/mu_w_face *
      model.gravity()*k_face[2]*face_normal[2]*face_area;

    // return T;

  } // eom

}  // end of namespace

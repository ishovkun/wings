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
    // update all values and wells
    virtual void update(const CellIterator<dim> &cell,
                        const double pressure,
                        const std::vector<double> &extra_values,
                        const bool update_well=true);
    // light version of the previous function - dosn't update wells
    // and values from extra_values
    virtual double get_mass_matrix_entry() const;
    virtual void update_face_values(const CellValuesBase<dim> &neighbor_data,
                                    const Tensor<1,dim>       &dx,
                                    const Tensor<1,dim>       &face_normal,
                                    const double              dS);


   public:
    double Q, J, T_face, G_face;
   protected:
    const Model::Model<dim> &model;
    double                   phi, mu_w, B_w, C_w, cell_volume;
    Point<dim>               cell_coord;
    Vector<double>           k;
    std::vector<double>      pvt_values_water;
  };


  template <int dim>
  CellValuesBase<dim>::CellValuesBase(const Model::Model<dim> &model_)
    :
    model(model_),
    k(dim),
    pvt_values_water(model.n_pvt_water_columns - 1) // cause p not really an entry
  {}



  template <int dim>
  void
  CellValuesBase<dim>::update(const CellIterator<dim> &cell,
                              const double pressure,
                              const std::vector<double> &extra_values,
                              const bool update_well)
  {
    AssertThrow(extra_values.size() == 0,
                ExcDimensionMismatch(extra_values.size(), 0));
    // PVT
    model.get_pvt_water(pressure, pvt_values_water);
    B_w = pvt_values_water[0];
    C_w = pvt_values_water[1];
    mu_w = pvt_values_water[2];

    //
    cell_coord = cell->center();
    phi = model.get_porosity->value(cell_coord);
    model.get_permeability->vector_value(cell_coord, k);
    cell_volume = cell->measure();
    // calculate source term
    Q = 0;
    J = 0;
    if (update_well)
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



  template <int dim>
  class CellValuesMP : public CellValuesBase<dim>
  {
   public:
    CellValuesMP(const Model::Model<dim> &model_);
    virtual void update(const CellIterator<dim> &cell,
                        const double pressure,
                        const std::vector<double> &extra_values,
                        const bool update_well=1);
    virtual void update_face_values(const CellValuesMP<dim> &neighbor_data,
                                    const Tensor<1,dim>     &dx,
                                    const Tensor<1,dim>     &face_normal,
                                    const double            dS);
    double get_mass_matrix_entry() const;

   protected:
    std::vector<double> pvt_values_oil, pvt_values_gas;
    double p;
    double mu_o, B_o, C_o,
           mu_g, B_g, C_g;
    double Sw, So, Sg;
    double krw, kro, krg;
    double
      c1w, c1p, c1e,
      c2o, c2p, c2e,
      c3g, c3o, c3w, c3p, c3e;
    std::vector<double> rel_perm;
    Vector<double>      vector_J_phase;
    Vector<double>      vector_Q_phase;
    Vector<double>      saturation;
  };



template <int dim>
CellValuesMP<dim>::CellValuesMP(const Model::Model<dim> &model_)
    :
    CellValuesBase<dim>::CellValuesBase(model_),
    pvt_values_oil(model_.n_pvt_oil_columns - 1), // cause p not really an entry
    pvt_values_gas(model_.n_pvt_gas_columns - 1), // cause p not really an entry
    rel_perm(model_.n_phases()),
    vector_J_phase(model_.n_phases()),
    vector_Q_phase(model_.n_phases()),
    saturation(model_.n_phases())
{}



template <int dim>
void
CellValuesMP<dim>::update(const CellIterator<dim> &cell,
                          const double pressure,
                          const std::vector<double> &extra_values,
                          const bool update_well)
{
  const auto & model = this->model;
  AssertThrow(extra_values.size() == model.n_phases()-1,
              ExcDimensionMismatch(extra_values.size(),
                                   model.n_phases()-1));

  this->phi = model.get_porosity->value(cell->center());
  model.get_permeability->vector_value(cell->center(), this->k);
  this->cell_volume = cell->measure();
  p = pressure;

  // Phase-dependent values
  if (model.has_phase(Model::Phase::Water))
  {
    auto & pvt_values_water = this->pvt_values_water;
    model.get_pvt_water(pressure, pvt_values_water);
    // std::cout << "got water" << std::endl;
    this->B_w = pvt_values_water[0];
    this->C_w = pvt_values_water[1];
    this->mu_w = pvt_values_water[2];

    // std::cout << "mu_w " << this->mu_w << std::endl;
    // std::cout << "B_w " << this->B_w << std::endl;
    // std::cout << "c_w " << this->C_w << std::endl;

  }
  if (model.has_phase(Model::Phase::Oil))
  {
    // std::cout << "getting oil" << std::endl;
    // std::cout << "size " << pvt_values_oil.size() << std::endl;
    model.get_pvt_oil(pressure, pvt_values_oil);
    B_o = pvt_values_oil[0];
    C_o = pvt_values_oil[1];
    mu_o = pvt_values_oil[2];
  }

  // water coeffs
  Sw = extra_values[0];
  c1w = this->phi /this->B_w * this->cell_volume;
  c1p = Sw / this->B_w * this->phi * this->C_w;
  c1e = 0;

  // oil coeffs
  So = 1.0 - Sw;
  // if (model.has_phase(Model::Phase::Oil) ||
  //     !model.has_phase(Model::Phase::Gas))
  //   So = 1.0 - Sw;
  // if (model.has_phase(Model::Phase::Oil) ||
  //     model.has_phase(Model::Phase::Gas))
  //   So = extra_values[1];

  c2o = this->phi / B_o * this->cell_volume;
  c2p = So * this->phi * C_o * this->cell_volume;
  c2e = 0;
  // gas coeffs
  // double Sg = 0;
  // if (!model.has_phase(Model::Phase::Oil) ||
  //     model.has_phase(Model::Phase::Gas))
  //   Sg = 1.0 - Sw;
  // if (model.has_phase(Model::Phase::Oil) ||
  //     model.has_phase(Model::Phase::Gas))
  //   So = 1.0 - Sw - So;
  // c3g = this->phi / B_g * this->cell_volume;
  // c3o = Rgo*c2o;
  // c3w = Rgw*c1w;

  c3g = 0;
  c3o = 0;
  c3w = 0;
  c3p = 0;
  c3e = 0;

  saturation[0] = Sw;
  saturation[1] = So;
  model.get_relative_permeability(saturation, rel_perm);

  // calculate source term
  this->Q = 0;
  this->J = 0;
  if (update_well)
  {
    vector_J_phase = 0;
    vector_Q_phase = 0;

    for (unsigned int phase = 0; phase<this->model.n_phases(); ++phase)
      for (const auto & well : model.wells)
      {
        std::pair<double,double> J_and_Q = well.get_J_and_Q(cell, phase);
        vector_J_phase[phase] += J_and_Q.first;
        vector_Q_phase[phase] += J_and_Q.second;
      }

    const double Qw = vector_Q_phase[0];
    const double Qo = vector_Q_phase[1];
    const double Jw = vector_J_phase[0];
    const double Jo = vector_J_phase[1];

    this->Q = Qw + B_o/this->B_w*Qo;
    this->J = Jw + B_o/this->B_w*Jo;
  }  // end update well
} // eom



template <int dim>
void
CellValuesMP<dim>::
update_face_values(const CellValuesMP<dim> &neighbor_data,
                   const Tensor<1,dim>     &dx,
                   const Tensor<1,dim>     &face_normal,
                   const double            face_area)
{
  const auto & model = this->model;
  this->T_face = 0;
  this->G_face = 0;

  Vector<double> k_face(3);
  Math::harmonic_mean(this->k, neighbor_data.k, k_face);
  const double mu_w_face = Math::arithmetic_mean(this->mu_w, neighbor_data.mu_w);
  const double B_w_face = Math::arithmetic_mean(this->B_w, neighbor_data.B_w);
  const double mu_o_face = Math::arithmetic_mean(mu_o, neighbor_data.mu_o);
  const double B_o_face = Math::arithmetic_mean(B_o, neighbor_data.B_o);

  // potential for upwinding
  const double pot_w =
      p + model.density_sc_water()/this->B_w * model.gravity();
  const double pot_w_neighbor =
      neighbor_data.p +
      model.density_sc_water()/neighbor_data.B_w*model.gravity();

  const double pot_o =
      p + model.density_sc_oil() / B_o * model.gravity();
  const double pot_o_neighbor =
      neighbor_data.p +
      model.density_sc_oil()/neighbor_data.B_o*model.gravity();

  // upwind relperms
  const double k_rw_face = Math::upwind(rel_perm[0], neighbor_data.rel_perm[0],
                                        pot_w, pot_w_neighbor);
  const double k_ro_face = Math::upwind(rel_perm[1], neighbor_data.rel_perm[1],
                                        pot_o, pot_o_neighbor);

  const double distance = dx.norm(); // to normalize
  // const double distance = (neighbor_data.cell_coord - this->cell_coord).norm();
  // const double distance =
  if (distance == 0.0)
    return;

  double Tw_face = 0;
  double To_face = 0;
  for (int d=0; d<dim; ++d)
  {
    if (abs(dx[d]/distance) > DefaultValues::small_number)
      {
        Tw_face += 1./mu_w_face/B_w_face *
            (k_face[d]*k_rw_face*face_normal[d]/dx[d])*face_area;
        To_face += 1./mu_o_face/B_o_face *
            (k_face[d]*k_ro_face*face_normal[d]/dx[d])*face_area;
      }
  }

  this->T_face = -c2o/c1w * Tw_face + To_face;

  // // G_face = data.density_sc_water()/B_w_face*data.gravity()*T_face*dx[2]*face_normal[2];
  // G_face = model.density_sc_water()/B_w_face/B_w_face/mu_w_face *
  //   model.gravity()*k_face[2]*face_normal[2]*face_area;

  const double Gw_face = model.density_sc_water()/B_w_face/B_w_face/mu_w_face *
      model.gravity()*k_face[2]*k_rw_face*face_normal[2]*face_area;
  const double Go_face = model.density_sc_oil()/B_o_face/B_o_face/mu_o_face *
      model.gravity()*k_face[2]*k_ro_face*face_normal[2]*face_area;

  this->G_face = -c2o/c1w * Gw_face + Go_face;


  // // return T;

} // eom



template <int dim>
double
CellValuesMP<dim>::get_mass_matrix_entry() const
{
  const double A = -c2o/c1w;
  const double B = 0;
  // A = c2o/c1w * (c3g-c3w)/(c3g-c3o);
  // B = c20 / (c3g - c3o);
  const double B_mass = A*c1p + c2p + B*c3p;
  return B_mass;
}


}  // end of namespace

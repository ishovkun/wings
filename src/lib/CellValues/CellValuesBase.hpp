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
    virtual void update(const CellIterator<dim> &cell,
                        const double pressure,
                        const std::vector<double> &extra_values);
    // this method is to later get J and Q
    virtual void update_wells(const CellIterator<dim> &cell);
    // this method computes true Q given the current cell block pressure
    virtual void update_wells(const CellIterator<dim> &cell,
                              const double pressure);
    virtual void update_face_values(const CellValuesBase<dim> &neighbor_data,
                                    const Tensor<1,dim>       &face_normal,
                                    const double               dS);
    double get_mass_matrix_entry() const;
    double get_J() const;
    double get_Q() const;
    double get_T_face() const;
    double get_G_face() const;

   public:
    const Model::Model<dim> & model;
    Vector<double>            k;
    std::vector<double>       rel_perm;
    Vector<double>            saturation;
    Point<dim>                cell_coord;
    std::vector<double>       pvt_values_water,
                              pvt_values_oil,
                              pvt_values_gas;
    // for wells
    Vector<double>            vector_J_phase;
    Vector<double>            vector_Q_phase;
    // coeffs
    double pressure;
    double phi, cell_volume;
    double mu_w, B_w, C_w,
           mu_o, B_o, C_o,
           mu_g, B_g, C_g,
           c1w, c1p, c1e,
           c2o, c2p, c2e,
           c3g, c3o, c3w, c3p, c3e;
    double T_w_face, T_o_face, T_g_face;
    double G_w_face, G_o_face, G_g_face;
    double Sw, So, Sg;

  };



template <int dim>
CellValuesBase<dim>::CellValuesBase(const Model::Model<dim> &model_)
    :
    model(model_),
    k(dim),
    rel_perm(model.n_phases()),
    saturation(model.n_phases()),
    pvt_values_water(model.n_pvt_water_columns - 1), // cause p not really an entry
    pvt_values_oil(model.n_pvt_oil_columns - 1), // cause p not really an entry
    pvt_values_gas(model.n_pvt_gas_columns - 1), // cause p not really an entry
    vector_J_phase(model.n_phases()),
    vector_Q_phase(model.n_phases())
{}



template <int dim>
void
CellValuesBase<dim>::update(const CellIterator<dim> &cell,
                            const double pressure,
                            const std::vector<double> &extra_values)
{
  const auto & model = this->model;
  AssertThrow(extra_values.size() == model.n_phases()-1,
              ExcDimensionMismatch(extra_values.size(),
                                   model.n_phases()-1));

  this->cell_coord = cell->center();
  this->phi = model.get_porosity->value(cell->center());
  model.get_permeability->vector_value(cell->center(), this->k);
  this->cell_volume = cell->measure();
  this->pressure = pressure;

  // determine saturations
  this->Sw = 0;
  this->So = 0;
  this->Sg = 0;

  if (model.has_phase(Model::Phase::Water))
  {
    if (model.n_phases() > 1)
    {
      this->Sw = extra_values[0];
      saturation[0] = this->Sw;
    }
    else
    {
      this->Sw = 1.0;
    }
  }

  if (model.has_phase(Model::Phase::Oil))
  {
    if (model.type == Model::ModelType::WaterOil)
      this->So = 1.0 - this->Sw;
    else if (model.type == Model::ModelType::Blackoil)
      this->So = extra_values[1];
    saturation[1] = this->So;
  }

  if (model.has_phase(Model::Phase::Gas))
  {
    this->Sg = 1 - this->Sw - this->So;
    if (model.type == Model::ModelType::WaterGas)
      saturation[1] = this->Sg;
    else if (model.type == Model::ModelType::Blackoil)
      saturation[2] = this->Sg;
  }

  // Phase-dependent values
  if (model.has_phase(Model::Phase::Water))
  {
    model.get_pvt_water(pressure, pvt_values_water);
    this->B_w = pvt_values_water[0];
    this->C_w = pvt_values_water[1];
    this->mu_w = pvt_values_water[2];

    c1w = this->phi * this->cell_volume / this->B_w;
    c1p = this->phi * this->Sw * this->C_w * this->cell_volume / this->B_w ;
    c1e = 0;
    // std::cout << "phi = "<< this->phi << std::endl;
    // std::cout << "Bw = "<< this->B_w << std::endl;
    // std::cout << "Cw = "<< this->C_w << std::endl;
    // std::cout << "mu_w = "<< this->mu_w << std::endl;
    // std::cout << "S_w = "<< this->Sw << std::endl;
    // std::cout << "V = "<< this->cell_volume << std::endl;
  }
  if (model.has_phase(Model::Phase::Oil))
  {
    model.get_pvt_oil(pressure, pvt_values_oil);
    this->B_o = pvt_values_oil[0];
    this->C_o = pvt_values_oil[1];
    this->mu_o = pvt_values_oil[2];
    // std::cout << "Bo = "<< this->B_o << std::endl;
    // std::cout << "Co = "<< this->C_o << std::endl;
    // std::cout << "mu_o = "<< this->mu_o << std::endl;
    // std::cout << "S_o = "<< this->So << std::endl;

    c2o = this->phi * this->cell_volume / this->B_o;
    c2p = this->phi * So * this->C_o * this->cell_volume / this->B_o;
    c2e = 0;
  }
  if (model.has_phase(Model::Phase::Gas))
  {
    AssertThrow(false, ExcNotImplemented());
    // c3p = 0;
    // c3e = 0;
    // c3w = Rgw*c1w;
    // c3g = this->phi / B_g * this->cell_volume;
    // c3o = Rgo*c2o;
  }

  // Rel perm
  if (model.n_phases() == 1)
    this->rel_perm[0] = 1;
  if (model.n_phases() == 2)
    model.get_relative_permeability(saturation, this->rel_perm);

} // eom



template <int dim>
void
CellValuesBase<dim>::update_wells(const CellIterator<dim> &cell)
{
  vector_J_phase = 0;
  vector_Q_phase = 0;

  for (unsigned int phase = 0; phase < this->model.n_phases(); ++phase)
    for (const auto & well : model.wells)
    {
      std::pair<double,double> J_and_Q = well.get_J_and_Q(cell, phase);
      vector_J_phase[phase] += J_and_Q.first;
      vector_Q_phase[phase] += J_and_Q.second;
    }

}  // eom



template <int dim>
void
CellValuesBase<dim>::
update_wells(const CellIterator<dim> &cell,
             const double             pressure)
{
  vector_Q_phase = 0;

  for (unsigned int phase = 0; phase < this->model.n_phases(); ++phase)
    for (const auto & well : model.wells)
      vector_Q_phase[phase] += well.get_flow_rate(cell, pressure, phase);
}  // eom



template <int dim>
void
CellValuesBase<dim>::
update_face_values(const CellValuesBase<dim> &neighbor_data,
                   const Tensor<1,dim>     &face_normal,
                   const double            face_area)
{
  const auto & model = this->model;

  Vector<double> k_face(dim);
  Math::harmonic_mean(this->k, neighbor_data.k, k_face);

  // geometric data
  const auto & dx = (neighbor_data.cell_coord - this->cell_coord);
  const double distance = dx.norm();
  // AssertThrow(distance > DefaultValues::small_number,
  //             ExcMessage("Cells too close"));

  // face relative permeabilities
  if (model.has_phase(Model::Phase::Water))
  {
    const double mu_w_face = Math::arithmetic_mean(this->mu_w, neighbor_data.mu_w);
    const double B_w_face = Math::arithmetic_mean(this->B_w, neighbor_data.B_w);
    // potential for upwinding
    const double pot_w =
        this->pressure + model.density_sc_water()/this->B_w * model.gravity();
    const double pot_w_neighbor =
        neighbor_data.pressure +
        model.density_sc_water()/neighbor_data.B_w*model.gravity();
    // upwind relperms
    const double k_rw_face = Math::upwind(this->rel_perm[0],
                                          neighbor_data.rel_perm[0],
                                          pot_w, pot_w_neighbor);

    T_w_face = 0;
    for (int d=0; d<dim; ++d)
      if (abs(dx[d]/distance) > DefaultValues::small_number)
        T_w_face += 1./mu_w_face/B_w_face *
            (k_face[d]*k_rw_face*face_normal[d]/dx[d])*face_area;

    G_w_face = model.density_sc_water()/B_w_face/B_w_face/mu_w_face *
        model.gravity()*k_face[2]*k_rw_face*face_normal[2]*face_area;
  }

  if (model.has_phase(Model::Phase::Oil))
  {
    const double mu_o_face = Math::arithmetic_mean(this->mu_o, neighbor_data.mu_o);
    const double B_o_face = Math::arithmetic_mean(this->B_o, neighbor_data.B_o);
    // potential for upwinding
    const double pot_o =
        this->pressure + model.density_sc_oil() / this->B_o * model.gravity();
    const double pot_o_neighbor =
        neighbor_data.pressure +
        model.density_sc_oil()/neighbor_data.B_o*model.gravity();
    // upwind relperms
    const double k_ro_face = Math::upwind(this->rel_perm[1],
                                          neighbor_data.rel_perm[1],
                                          pot_o, pot_o_neighbor);
    T_o_face = 0;
    for (int d=0; d<dim; ++d)
      if (abs(dx[d]/distance) > DefaultValues::small_number)
        T_o_face += 1./mu_o_face/B_o_face *
            (k_face[d]*k_ro_face*face_normal[d]/dx[d])*face_area;
    G_o_face = model.density_sc_oil()/B_o_face/B_o_face/mu_o_face *
        model.gravity()*k_face[2]*k_ro_face*face_normal[2]*face_area;
  }

  if (model.has_phase(Model::Phase::Gas))
  {
    AssertThrow(false, ExcNotImplemented());
    T_g_face = 0;
  }

} // eom



template <int dim>
inline
double
CellValuesBase<dim>::get_J() const
{
  double J = 0;
  if (model.type == Model::ModelType::SingleLiquid)
    J = vector_J_phase[0];
  else if (model.type == Model::ModelType::WaterOil)
    J = vector_J_phase[0] + this->B_o/this->B_w*vector_J_phase[1];
  else
    AssertThrow(false, ExcNotImplemented());

  return J;
}



template <int dim>
inline
double
CellValuesBase<dim>::get_Q() const
{
  double Q = 0;
  if (model.type == Model::ModelType::SingleLiquid)
    Q = vector_Q_phase[0];
  else if (model.type == Model::ModelType::WaterOil)
    Q = vector_Q_phase[0] + this->B_o/this->B_w*vector_Q_phase[1];
  else
    AssertThrow(false, ExcNotImplemented());

  return Q;
}



template <int dim>
inline
double
CellValuesBase<dim>::get_T_face() const
{
  double T_face = 0;
  if (model.type == Model::ModelType::SingleLiquid)
    T_face = T_w_face;
  else if (model.type == Model::ModelType::WaterOil)
    T_face = -c2o/c1w * T_w_face + T_o_face;
  else
    AssertThrow(false, ExcNotImplemented());

  return T_face;
}



template <int dim>
inline
double
CellValuesBase<dim>::get_G_face() const
{
  double G_face = 0;
  if (model.type == Model::ModelType::SingleLiquid)
    G_face = G_w_face;
  else if (model.type == Model::ModelType::WaterOil)
    G_face = -c2o/c1w * G_w_face + G_o_face;
  else
    AssertThrow(false, ExcNotImplemented());

  return G_face;
}


template <int dim>
double
CellValuesBase<dim>::get_mass_matrix_entry() const
{
  double B_mass = 0;
  const auto & model = this->model;
  if (model.type == Model::ModelType::SingleLiquid)
  {
    // return cell_volume/this->B_w*(phi*C_w);
    B_mass = c1p;
  }
  else if (model.type == Model::ModelType::WaterOil)
  {
    // std::cout << "c1p " << c1p << "\t" << std::endl;
    // std::cout << "c1w " << c1w << "\t" << std::endl;
    // std::cout << "c2p " << c2p << "\t" << std::endl;
    // std::cout << "c2o " << c2o << "\t" << std::endl;
    // std::cout << "B " << c2o/c1w*c1p + c2p << "\t" << std::endl;

    const double A = +c2o/c1w;
    B_mass = A*c1p + c2p;
  }
  else if (model.type == Model::ModelType::Blackoil)
  {
    const double A = c2o/c1w * (c3g-c3w)/(c3g-c3o);
    const double B = c2o / (c3g - c3o);
    B_mass = A*c1p + c2p + B*c3p;
  }
  return B_mass;
}


}  // end of namespace

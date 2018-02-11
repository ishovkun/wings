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
    /* Update storage vectors and values for the current cell */
    virtual void update(const CellIterator<dim> &cell,
                        const double pressure,
                        const std::vector<double> &extra_values);
    /* Update wellbore rates and j-indices.
     * The calculated rates are not true rates for
     * pressure-controled wells,
     * Q-vector rather gets the value j_ind*BHP
     */
    virtual void update_wells(const CellIterator<dim> &cell);
    /* Update wellbore rates.
     * This method actually gets real rates for both flow- and pressure-
     * controlled wellbores.
     */
    virtual void update_wells(const CellIterator<dim> &cell,
                              const double pressure);
    /* Update storage vectors and values for the current face */
    virtual void update_face_values(const CellValuesBase<dim> &neighbor_data,
                                    const Tensor<1,dim>       &face_normal,
                                    const double               dS);
    // methods for pressure solver
    /* Get a matrix entry corresponding to the cell.
     * should be called once after update_values()
     */
    virtual double get_matrix_cell_entry(const double time_step) const;
    /* Get a rhs entry corresponding to the cell.
     * should be called once after update_values()
     */
    virtual double get_rhs_cell_entry(const double time_step,
                                      const double pressure,
                                      const double old_pressure,
                                      const int /* phase */ = 0) const;
    /* Get a matrix entry corresponding to the cell.
     * should be called once after update_values()
     */
    virtual double get_matrix_face_entry() const;
    /* Get a rhs entry corresponding to the face.
     * should be called once per face after update_face_values()
     */
    virtual double get_rhs_face_entry(const double /* time_step */,
                                      const int /* phase */ = 0) const;

   public:
    const Model::Model<dim> & model;      // reference to the model object
    Vector<double>            k;  // absolute permeability
    std::vector<double>       rel_perm;  // relative permeabilities
    Vector<double>            saturation;  // phase saturations
    Point<dim>                cell_coord;  // cell center coordinates
    // these vectors store current pvt values for phases
    std::vector<double>       pvt_values_water,
                              pvt_values_oil,
                              pvt_values_gas;
    // for wells
    Vector<double>            vector_J_phase;  // productivity indices for phases and segments
    Vector<double>            vector_Q_phase;  // well rates for all segments and phases
    // coeffs
    double pressure;  // stores current cell pressure
    double phi, cell_volume;  // current permeability and cell volume
    double mu_w, B_w, C_w,    // phase viscosities, volume factors, and compressiblities
           mu_o, B_o, C_o,
           mu_g, B_g, C_g,
           c1w, c1p, c1e,     // eq coeffs, see equations
           c2o, c2p, c2e,
           c3g, c3o, c3w, c3p, c3e;
    double T_w_face, T_o_face, T_g_face;  // cell phase transmissibilities
    double G_w_face, G_o_face, G_g_face;  // cell phase gravity vectors
    double Sw, So, Sg;                    // cell saturations

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
    if (model.fluid_model == Model::FluidModelType::DeadOil)
      this->So = 1.0 - this->Sw;
    else if (model.fluid_model == Model::FluidModelType::Blackoil)
      this->So = extra_values[1];
    saturation[1] = this->So;
  }

  if (model.has_phase(Model::Phase::Gas))
  {
    this->Sg = 1 - this->Sw - this->So;
    if (model.fluid_model == Model::FluidModelType::WaterGas)
      saturation[1] = this->Sg;
    else if (model.fluid_model == Model::FluidModelType::Blackoil)
      saturation[2] = this->Sg;
  }

  // Phase-dependent values
  if (model.has_phase(Model::Phase::Water))
  {
    model.get_pvt_water(pressure, pvt_values_water);
    this->B_w = pvt_values_water[0];
    this->C_w = pvt_values_water[1];
    this->mu_w = pvt_values_water[2];

    c1w = this->phi * this->cell_volume / this->B_w; // = d12
    c1p = this->phi * this->Sw * this->C_w * this->cell_volume / this->B_w ; // = d11
    c1e = 0;
    // std::cout << "phi = "<< this->phi << std::endl;
    // std::cout << "Bw = "<< this->B_w << std::endl;
    // std::cout << "Cw = "<< this->C_w << std::endl;
    // std::cout << "mu_w = "<< this->mu_w << std::endl;
    // std::cout << "S_w = "<< this->Sw << std::endl;
    // std::cout << "V = "<< this->cell_volume << std::endl;
    // if (cell->center()[0] < 1.5)
    // {
    //   std::cout << "c1p = "<< c1p << std::endl;
    //   std::cout << "c1w = "<< c1w << std::endl;
    //   std::cout << "c1p/c1w = "<< c1p/c1w << std::endl;
    // }

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
    {
      vector_Q_phase[phase] += well.get_flow_rate(cell, pressure, phase);

      // if (cell->center()[0] < 1.5)
      //   std::cout << "Q" << phase << " = " << vector_Q_phase[phase] << std::flush << std::endl;
    }
}  // eom



template <int dim>
void
CellValuesBase<dim>::
update_face_values(const CellValuesBase<dim> &neighbor_data,
                   const Tensor<1,dim>     &face_normal,
                   const double            face_area)
{
  const auto & model = this->model;

  // geometric data
  const auto & dx = (neighbor_data.cell_coord - this->cell_coord);
  const double distance = dx.norm();

  // obtain face absolute transmissibility
  Vector<double> k_face(dim);
  // Math::harmonic_mean(this->k, neighbor_data.k, k_face);
  // dirty hack to make harmonic mean work with irregular grid
  const double dx1 = cell_volume/face_area;
  const double dx2 = 2*(distance - dx1/2);
  Math::harmonic_mean(this->k, neighbor_data.k, dx1, dx2, k_face);

  double T_abs_face = 0;
  for (int d=0; d<dim; ++d)
    if (abs(dx[d]/distance) > DefaultValues::small_number)
      T_abs_face += (k_face[d]*abs(face_normal[d]/dx[d]))*face_area;


  // AssertThrow(distance > DefaultValues::small_number,
  //             ExcMessage("Cells too close"));

  // face relative permeabilities
  if (model.has_phase(Model::Phase::Water))
  {
    const double mu_w_face = Math::arithmetic_mean(this->mu_w, neighbor_data.mu_w);
    const double B_w_face = Math::arithmetic_mean(this->B_w, neighbor_data.B_w);
    // potential for upwinding
    const double pot_w =
        this->pressure
        +
        model.density_sc_water()/this->B_w * model.gravity() *
        this->cell_coord[2];
    const double pot_w_neighbor =
        neighbor_data.pressure +
        model.density_sc_water()/neighbor_data.B_w*model.gravity() *
        this->cell_coord[2];
    // upwind relperms
    const double k_rw_face = Math::upwind(this->rel_perm[0],
                                          neighbor_data.rel_perm[0],
                                          pot_w, pot_w_neighbor);
    // if (cell_coord[0] < 1.5)
    // {
    //   std:: cout << "Sw = " << this->Sw << std::endl;
    //   std::cout << "krw" << " = " << k_rw_face << std::flush << std::endl;
    // }

    T_w_face = T_abs_face*k_rw_face/mu_w_face/B_w_face;
    // for (int d=0; d<dim; ++d)
    //   if (abs(dx[d]/distance) > DefaultValues::small_number)
    //     T_w_face += 1./mu_w_face/B_w_face *
    //         (k_face[d]*k_rw_face*abs(face_normal[d]/dx[d]))*face_area;

    G_w_face = model.density_sc_water()/B_w_face/B_w_face/mu_w_face *
        model.gravity()*k_face[2]*k_rw_face*face_normal[2]*face_area;
  }

  if (model.has_phase(Model::Phase::Oil))
  {
    const double mu_o_face = Math::arithmetic_mean(this->mu_o, neighbor_data.mu_o);
    const double B_o_face = Math::arithmetic_mean(this->B_o, neighbor_data.B_o);

    // potential for upwinding
    const double pot_o =
        this->pressure
        +
        model.density_sc_oil() / this->B_o * model.gravity() *
        this->cell_coord[2];

    const double pot_o_neighbor =
        neighbor_data.pressure
        +
        model.density_sc_oil()/neighbor_data.B_o*model.gravity() *
        this->cell_coord[2];
    // upwind relperms
    const double k_ro_face = Math::upwind(this->rel_perm[1],
                                          neighbor_data.rel_perm[1],
                                          pot_o, pot_o_neighbor);

    T_o_face = T_abs_face*k_ro_face/mu_o_face/B_o_face;
    // for (int d=0; d<dim; ++d)
    //   if (abs(dx[d]/distance) > DefaultValues::small_number)
    //     T_o_face += 1./mu_o_face/B_o_face *
    //         (k_face[d]*k_ro_face*abs(face_normal[d]/dx[d]))*face_area;
    G_o_face = model.density_sc_oil()/B_o_face/B_o_face/mu_o_face *
        model.gravity()*k_face[2]*k_ro_face*face_normal[2]*face_area;

    // if (cell_coord[0] < 1.5)
    // if (cell_coord[0] > 1.5 && cell_coord[0] < 3.0)
    // {
    //   std::cout << "i = " << 1 << std::endl;
    //   std::cout << "neighbor = " << neighbor_data.cell_coord[0] << std::endl;
    //   std:: cout << "So = " << this->So << std::endl;
    //   std::cout << "kro" << " = " << k_ro_face << std::flush << std::endl;
    //   std::cout << "face_area" << " = " << face_area << std::endl;
    //   std::cout << "face_normal" << " = " << face_normal << std::endl;
    //   std::cout << "To_face" << " = " << T_o_face << std::endl;
    //   std::cout << "k_face" << " = " << k_face << std::endl;
    // }
  }

  if (model.has_phase(Model::Phase::Gas))
  {
    AssertThrow(false, ExcNotImplemented());
    T_g_face = 0;
  }

} // eom



template <int dim>
double
CellValuesBase<dim>::
get_matrix_cell_entry(const double time_step) const
{
  double entry = 0;
  const auto & model = this->model;
  if (model.fluid_model == Model::FluidModelType::Liquid)
  {
    // B_mass = c1p;
    // J = vector_J_phase[0];
    entry += c1p/time_step;
    entry += vector_J_phase[0];
  }
  else if (model.fluid_model == Model::FluidModelType::DeadOil)
  {
    // B_mass = c2o/c1w * c1p + c2p;
    // J = +c2o/c1w*vector_J_phase[0] + vector_J_phase[1];
    entry += (c2o/c1w * c1p + c2p)/time_step;
    entry += +c2o/c1w*vector_J_phase[0] + vector_J_phase[1];

  }
  else if (model.fluid_model == Model::FluidModelType::Blackoil)
  {
    const double A = c2o/c1w * (c3g-c3w)/(c3g-c3o);
    const double B = c2o / (c3g - c3o);
    // B_mass = A*c1p + c2p + B*c3p;
    entry += (A*c1p + c2p + B*c3p)/time_step;

    // need to get equations for J
    AssertThrow(false, ExcNotImplemented());
  }
  return entry;
} // eom



template <int dim>
double
CellValuesBase<dim>::
get_rhs_cell_entry(const double time_step,
                   const double pressure,
                   const double,
                   const int) const
{
  // two last variables are ignored (used in children)

  double entry = 0;
  const auto & model = this->model;
  if (model.fluid_model == Model::FluidModelType::Liquid)
  {
    // B_mass = c1p;
    // J = vector_J_phase[0];
    entry += c1p * pressure/time_step; // B matrix
    entry += vector_Q_phase[0];  // Q vector
  }
  else if (model.fluid_model == Model::FluidModelType::DeadOil)
  {
    // B_mass = c2o/c1w * c1p + c2p;
    // J = +c2o/c1w*vector_J_phase[0] + vector_J_phase[1];
    entry += (c2o/c1w * c1p + c2p)*pressure/time_step;  // B matrix
    entry += +c2o/c1w*vector_Q_phase[0] + vector_Q_phase[1]; // Q vector

  }
  else if (model.fluid_model == Model::FluidModelType::Blackoil)
  {
    const double A = c2o/c1w * (c3g-c3w)/(c3g-c3o);
    const double B = c2o / (c3g - c3o);
    // B_mass = A*c1p + c2p + B*c3p;
    entry += (A*c1p + c2p + B*c3p)*pressure/time_step;  // B matrix

    // need to get equations for J and Q
    AssertThrow(false, ExcNotImplemented());
  }
  return entry;
} // eom



template <int dim>
inline
double
CellValuesBase<dim>::get_matrix_face_entry() const
{
  double entry = 0;
  if (model.fluid_model == Model::FluidModelType::Liquid)
    entry += T_w_face;
  else if (model.fluid_model == Model::FluidModelType::DeadOil)
  {
    entry += +c2o/c1w * T_w_face + T_o_face;
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return entry;
} // eom



template <int dim>
inline
double
CellValuesBase<dim>::get_rhs_face_entry(const double,
                                        const int) const
{
  double entry = 0;
  if (model.fluid_model == Model::FluidModelType::Liquid)
    entry += G_w_face;
  else if (model.fluid_model == Model::FluidModelType::DeadOil)
  {
    entry += +c2o/c1w * G_w_face + G_o_face;
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return entry;
} // eom

}  // end of namespace

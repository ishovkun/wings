#pragma once

#include <Model.hpp>
#include <Math.hpp>

#include <Equations/FluidEquationsBase.hpp>


namespace Equations
{
using namespace dealii;

// static const int dim = 3;

// class IMPESPressure : public FluidEquationsBase<dim>
template <int n_phases>
class IMPESPressure : public FluidEquationsBase
{
 public:
  IMPESPressure(const Model::Model<dim> &model_);
  /* Update storage vectors and values for the current cell */
  virtual void update_cell_values(const CellIterator<dim> & cell,
                                  const SolutionValues    & solution_values) override;
  /* Update storage vectors and values for the current face */
  virtual void update_face_values(const CellIterator<dim> & neighbor_cell,
                                  const SolutionValues    & solution_values,
                                  const FaceGeometry      & geometry) override;
  /* Update wellbore rates and j-indices.
   * The calculated rates are not true rates for
   * pressure-controled wells,
   * Q-vector rather gets the value j_ind*BHP
   */
  virtual void update_wells(const CellIterator<dim> &cell) override;
  /* Update wellbore rates.
   * This method actually gets real rates for both flow- and pressure-
   * controlled wellbores.
   */
  virtual void update_wells(const CellIterator<dim> & cell,
                            const double              pressure) override;
  // methods for pressure solver
  /* Get a matrix entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_matrix_cell_entry(const double time_step) const override;
  /* Get a rhs entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_rhs_cell_entry(const double time_step,
                                    const double pressure,
                                    const double old_pressure,
                                    const int /* phase */ = 0) const override;
  /* Get a matrix entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_matrix_face_entry(const int = 0) const override;
  /* Get a rhs entry corresponding to the face.
   * should be called once per face after update_face_values()
   */
  virtual double get_rhs_face_entry(const double /* time_step */,
                                    const int /* phase */ = 0) const override;

 public:
  const Model::Model<dim>     & model;        // reference to the model object
  Point<dim>                    location;     // cell center coordinates
  // Tensor<1,n_phases>            rel_perm;           // productivity indices for phases and segments
  SymmetricTensor<2,dim>        perm;
  std::vector<double>           rel_perm;              // relative permeabilities
  // Tensor<1,n_phases>            saturations;           // phase saturations
  Vector<double>                saturations;           // phase saturations
  std::vector<Model::PVTValues> pvt_values;            // phase volume factors
  Model::PVTValues              pvt_neighbor;          // phase volume factors
  Tensor<1,n_phases>            face_transmissibility; // phase face transmissibilities
  Tensor<1,n_phases>            face_gravity_terms;    // phase face gravity terms
  Tensor<1,n_phases>            well_Js;               // productivity indices for phases and segments
  Tensor<1,n_phases>            well_Qs;               // well rates for all segments and phases
  Point<dim>                    cell_coord;
  // coeffs
  double pressure;         // stores current cell pressure
  double porosity, cell_volume; // porosity and cell volume
  // the next tensor defines these terms
  //   c1w,   c1o=0, c1g=0,  // eq coeffs, see equations
  //   c1w=0, c2o,   c2g=0,
  //   c3w,   c3o,   c3g,
  Tensor<2,n_phases> saturation_terms;
  // c1p, c2p, c3p
  Tensor<1,n_phases> pressure_terms;
};


// ----------------------Partial specialization-------------------------------
// partial class specialization
// template <int dim>
// class IMPESPressure<1, dim> //: public FluidEquationsBase<dim>
// {
//  public:
//   virtual double get_matrix_cell_entry(const double time_step) const override;
//   virtual double get_matrix_face_entry(const int = 0) const override;
//   virtual double get_rhs_cell_entry(const double time_step,
//                                     const double pressure,
//                                     const double old_pressure,
//                                     const int /* phase */ = 0) const override;
//   virtual double get_rhs_face_entry(const double /* time_step */,
//                                     const int /* phase */ = 0) const override;
// };

// template <int dim>
// class IMPESPressure<2, dim> //: public FluidEquationsBase<dim>
// {
//  public:
//   virtual double get_matrix_cell_entry(const double time_step) const override;
//   virtual double get_matrix_face_entry(const int = 0) const override;
//   virtual double get_rhs_cell_entry(const double time_step,
//                                     const double pressure,
//                                     const double old_pressure,
//                                     const int /* phase */ = 0) const override;
//   virtual double get_rhs_face_entry(const double /* time_step */,
//                                     const int /* phase */ = 0) const override;
// };

// template <int dim>
// class IMPESPressure<3,dim> //: public FluidEquationsBase<dim>
// {
//  public:
//   virtual double get_matrix_cell_entry(const double time_step) const override;
//   virtual double get_matrix_face_entry(const int = 0) const override;
//   virtual double get_rhs_cell_entry(const double time_step,
//                                     const double pressure,
//                                     const double old_pressure,
//                                     const int /* phase */ = 0) const override;
//   virtual double get_rhs_face_entry(const double /* time_step */,
//                                     const int /* phase */ = 0) const override;
// };
// ---------------------------------------------------------------------------


template <int n_phases>
IMPESPressure<n_phases>::
IMPESPressure(const Model::Model<dim> &model)
    :
    FluidEquationsBase::FluidEquationsBase(),
    model(model),
    rel_perm(n_phases),
    pvt_values(n_phases)
{}



template <int n_phases>
void
IMPESPressure<n_phases>::
update_cell_values(const CellIterator<dim> & cell,
                   const SolutionValues    & solution)
{
  location = cell->center();
  cell_volume = cell->measure();
  porosity = model.get_porosity->value(location);
  perm = model.get_permeability->value(cell->center());

  this->pressure = solution.pressure;
  saturations = solution.saturation;
  // make it a new class
  // this->delta_div_u =
  //     solution.grad_u.sum() - solution.old_grad_u.sum();
  // const double C_r = model.get_rock_compressibility(cell->center());

  for (int phase=0; phase<n_phases; ++phase)
  {
    model.get_pvt(pressure, phase, pvt_values[phase]);

    saturation_terms[phase][phase] =
        porosity / pvt_values[phase].volume_factor * cell_volume;
    pressure_terms[phase] =
        saturations[phase]/pvt_values[phase].volume_factor *
        (porosity*pvt_values[phase].compressibility) * cell_volume;

    // old
    // c1w[phase] = porosity / pvt_values[phase].volume_factor * cell_volume;
    // c1p = Sw/B_w * (C_r + phi*C_w) * cell_volume; // = d11 in balhoff
    // c1e = this->Sw/this->B_w * model.get_biot_coefficient();
  }

  model.get_relative_permeability(saturations, rel_perm);

} // eom



template <int n_phases>
void
IMPESPressure<n_phases>::
update_wells(const CellIterator<dim> &cell)
{
  well_Js = 0;
  well_Qs = 0;

  for (int phase = 0; phase < n_phases; ++phase)
    for (const auto & well : model.wells)
    {
      std::pair<double,double> J_and_Q = well.get_J_and_Q(cell, phase);
      well_Js[phase] += J_and_Q.first;
      well_Qs[phase] += J_and_Q.second;
    }

}  // eom



template <int n_phases>
void
IMPESPressure<n_phases>::
update_wells(const CellIterator<dim> &cell,
             const double             pressure)
{
  well_Qs = 0;

  for (int phase = 0; phase < n_phases; ++phase)
    for (const auto & well : model.wells)
      well_Qs[phase] += well.get_flow_rate(cell, pressure, phase);
}  // eom



template <int n_phases>
void
IMPESPressure<n_phases>::
update_face_values(const CellIterator<dim> & neighbor_cell,
                   const SolutionValues    & neighbor_solution,
                   const FaceGeometry      & face_geometry)
{
  const auto & dx = (neighbor_cell->center() - this->cell_coord);
  const double distance = dx.norm();

  // obtain face absolute transmissibility
  SymmetricTensor<2,dim> perm_neighbor =
      model.get_permeability->value(neighbor_cell->center());
  // Math::harmonic_mean(this->k, neighbor_data.k, k_face);
  // dirty hack to make harmonic mean work with irregular grid
  const double dx1 = cell_volume/face_geometry.area;
  const double dx2 = 2*(distance - dx1/2);
  const SymmetricTensor<2,dim> k_face =
      Math::harmonic_mean(perm, perm_neighbor, dx1, dx2);

  // face absolute transmissibility
  double T_abs_face = 0;
  // const Tensor<1,dim> kn = scalar_production(k_face, face_geometry.normal);
  const Tensor<1,dim> kn = k_face * face_geometry.normal;
  for (int d=0; d<dim; ++d)
    if (abs(dx[d]/distance) > DefaultValues::small_number)
      T_abs_face += abs(kn[d] / dx[d]) * face_geometry.area;

  // face phase transmissibilities
  std::vector<double> rel_perm_neighbor(n_phases);
  model.get_relative_permeability(neighbor_solution.saturation, rel_perm_neighbor);

  const double gravity = model.gravity();
  const double depth = location[2];
  const double neighbor_depth = neighbor_cell->center()[2];
  for (int phase=0; phase<n_phases; ++phase)
  {
    model.get_pvt(neighbor_solution.pressure, phase, pvt_neighbor);
    const double rho_sc = model.density_standard_conditions(phase);

    // arithmetic average properties
    const double mu_face = Math::arithmetic_mean(pvt_values[phase].viscosity,
                                                 pvt_neighbor.viscosity);
    const double B_face = Math::arithmetic_mean(pvt_values[phase].volume_factor,
                                                pvt_neighbor.volume_factor);

    // potential for upwinding
    const double potential =
        pressure
        +
        rho_sc / pvt_values[phase].volume_factor * gravity * depth;
    const double potential_neighbor =
        neighbor_solution.pressure
        +
        rho_sc / pvt_neighbor.volume_factor * gravity * neighbor_depth;


    const double rel_perm_face =
        Math::upwind(rel_perm[phase], rel_perm_neighbor[phase],
                     potential, potential_neighbor);

    // transmissibility
    face_transmissibility[phase] = T_abs_face * rel_perm_face / mu_face / B_face;

    // gravity term
    const int z = 2;
    face_gravity_terms[phase] =
        rho_sc / (B_face*B_face) / mu_face *
        gravity * kn[z] * rel_perm_face *
        face_geometry.area;
    // upwind relperms
  } // end phase loop

} // eom


// ==================== Partial specialization ===============================
// --------------------------- 1 phase ---------------------------------------
template <>
double
IMPESPressure<1>::
get_matrix_cell_entry(const double time_step) const
{
  double entry = 0;

  entry += this->pressure_terms[0]/time_step;
  entry += this->well_Js[0];
  return entry;

  // // coupling with geomechanics
  // if (model.coupling_strategy() == Model::FluidCouplingStrategy::FixedStressSplit)
  // {
  //   const double alpha = model.get_biot_coefficient();
  //   const double E = model.get_young_modulus->value(this->cell_coord);
  //   const double nu = model.get_poisson_ratio->value(this->cell_coord);
  //   const double bulk_modulus = E/3.0/(1.0-2.0*nu);
  //   entry += cell_volume/B_w * alpha*alpha/bulk_modulus/time_step;
  // }
}


// template <int dim>
template <>
double
IMPESPressure<1>::
get_rhs_cell_entry(const double time_step,
                   const double pressure,
                   const double,
                   const int) const
{
  double entry = 0;

  const double c1p = this->saturation_terms[0][0];

  entry += c1p * pressure/time_step; // B matrix
  entry += this->well_Qs[0];  // Q vector
  // entry += -c1e*delta_div_u/time_step; // poroelastic

  return entry;
}  // eom



// template <int dim>
template <>
inline
double
IMPESPressure<1>::get_matrix_face_entry(const int) const
{
  return this->face_transmissibility[0];
}



// template <int dim>
template <>
inline
double
IMPESPressure<1>::get_rhs_face_entry(const double,
                                     const int) const
{
  return this->face_gravity_terms[0];
} // eom

// --------------------------- 2 phase ---------------------------------------

// template <int n_phases>
// void
// IMPESPressure<dim,n_phases>::
// update_face_values(const CellIterator<dim> & neighbor_cell,
//                    const SolutionValues    & solution_values,
//                    const FaceGeometry      & face_geometry)
// {
//   const auto & model = this->model;

//   // geometric data
//   const auto & dx = (neighbor_data.cell_coord - this->cell_coord);
//   const double distance = dx.norm();

//   // obtain face absolute transmissibility
//   Vector<double> k_face(dim);
//   // Math::harmonic_mean(this->k, neighbor_data.k, k_face);
//   // dirty hack to make harmonic mean work with irregular grid
//   const double dx1 = cell_volume/face_values.area;
//   const double dx2 = 2*(distance - dx1/2);
//   Math::harmonic_mean(this->k, neighbor_data.k, dx1, dx2, k_face);

//   double T_abs_face = 0;
//   for (int d=0; d<dim; ++d)
//     if (abs(dx[d]/distance) > DefaultValues::small_number)
//       T_abs_face += (k_face[d]*abs(face_values.normal[d]/dx[d]))*face_values.area;


//   // AssertThrow(distance > DefaultValues::small_number,
//   //             ExcMessage("Cells too close"));

//   // face relative permeabilities
//   if (model.has_phase(Model::Phase::Water))
//   {
//     const double mu_w_face = Math::arithmetic_mean(this->mu_w, neighbor_data.mu_w);
//     const double B_w_face = Math::arithmetic_mean(this->B_w, neighbor_data.B_w);
//     // potential for upwinding
//     const double pot_w =
//         this->pressure
//         +
//         model.density_sc_water()/this->B_w * model.gravity() *
//         this->cell_coord[2];
//     const double pot_w_neighbor =
//         neighbor_data.pressure +
//         model.density_sc_water()/neighbor_data.B_w*model.gravity() *
//         this->cell_coord[2];
//     // upwind relperms
//     const double k_rw_face = Math::upwind(this->rel_perm[0],
//                                           neighbor_data.rel_perm[0],
//                                           pot_w, pot_w_neighbor);
//     // if (cell_coord[0] < 1.5)
//     // {
//     //   std:: cout << "Sw = " << this->Sw << std::endl;
//     //   std::cout << "krw" << " = " << k_rw_face << std::flush << std::endl;
//     // }

//     T_w_face = T_abs_face*k_rw_face/mu_w_face/B_w_face;

//     G_w_face = model.density_sc_water()/B_w_face/B_w_face/mu_w_face *
//         model.gravity()*k_face[2]*k_rw_face*face_values.normal[2]*face_values.area;
//   }

//   if (model.has_phase(Model::Phase::Oil))
//   {
//     const double mu_o_face = Math::arithmetic_mean(this->mu_o, neighbor_data.mu_o);
//     const double B_o_face = Math::arithmetic_mean(this->B_o, neighbor_data.B_o);

//     // potential for upwinding
//     const double pot_o =
//         this->pressure
//         +
//         model.density_sc_oil() / this->B_o * model.gravity() *
//         this->cell_coord[2];

//     const double pot_o_neighbor =
//         neighbor_data.pressure
//         +
//         model.density_sc_oil()/neighbor_data.B_o*model.gravity() *
//         this->cell_coord[2];
//     // upwind relperms
//     const double k_ro_face = Math::upwind(this->rel_perm[1],
//                                           neighbor_data.rel_perm[1],
//                                           pot_o, pot_o_neighbor);

//     T_o_face = T_abs_face*k_ro_face/mu_o_face/B_o_face;

//     G_o_face = model.density_sc_oil()/B_o_face/B_o_face/mu_o_face *
//         model.gravity()*k_face[2]*k_ro_face*face_values.normal[2]*face_values.area;

//     // if (cell_coord[0] < 1.5)
//     // if (cell_coord[0] > 1.5 && cell_coord[0] < 3.0)
//     // {
//     //   std::cout << "i = " << 1 << std::endl;
//     //   std::cout << "neighbor = " << neighbor_data.cell_coord[0] << std::endl;
//     //   std:: cout << "So = " << this->So << std::endl;
//     //   std::cout << "kro" << " = " << k_ro_face << std::flush << std::endl;
//     //   std::cout << "face_area" << " = " << face_area << std::endl;
//     //   std::cout << "face_normal" << " = " << face_normal << std::endl;
//     //   std::cout << "To_face" << " = " << T_o_face << std::endl;
//     //   std::cout << "k_face" << " = " << k_face << std::endl;
//     // }
//   }

//   if (model.has_phase(Model::Phase::Gas))
//   {
//     AssertThrow(false, ExcNotImplemented());
//     T_g_face = 0;
//   }

// } // eom





// template <int n_phases>
// template <int dim>
// double
// IMPESPressure<2,dim>::
// get_matrix_cell_entry(const double time_step) const
// {
//   double entry = 0;

//   if (this->model.fluid_model == Model::FluidModelType::DeadOil)
//   {
//     const double c1w = this->saturation_terms[0][0];
//     const double c2o = this->saturation_terms[1][1];
//     const double c1p = this->pressure_terms[0];
//     const double c2p = this->pressure_terms[1];
//     entry += (c2o/c1w * c1p + c2p)/time_step;
//     entry += +c2o/c1w*this->well_Js[0] + this->well_Js[1];
//   }
//   else
//   {
//     AssertThrow(false, ExcNotImplemented());
//   }

//   return entry;
// } // eom



// template <int dim>
// double
// IMPESPressure<3,dim>::
// get_matrix_cell_entry(const double time_step) const
// {
//   AssertThrow(false, ExcNotImplemented());

//   const double entry = 0;
//   const double c1w = this->saturation_terms[0][0];
//   const double c2o = this->saturation_terms[1][1];
//   const double c3w = this->saturation_terms[2][0];
//   const double c3o = this->saturation_terms[2,1];
//   const double c3g = this->saturation_terms[2][2];
//   const double c1p = this->pressure_terms[0];
//   const double c2p = this->pressure_terms[1];
//   const double c3p = this->pressure_terms[2];

//   const double A = c2o/c1w * (c3g-c3w)/(c3g-c3o);
//   const double B = c2o / (c3g - c3o);

//   entry += (A*c1p + c2p + B*c3p)/time_step;

//   return entry;
// }


// template <int n_phases>
// double
// IMPESPressure<dim,n_phases>::
// get_matrix_cell_entry(const double time_step) const
// {
//   double entry = 0;
//   const auto & model = this->model;
//   if (model.fluid_model == Model::FluidModelType::Liquid)
//   {
//     // B_mass = c1p;
//     // J = well_Js[0];
//     entry += c1p/time_step;
//     entry += well_Js[0];
//   }
//   else if (model.fluid_model == Model::FluidModelType::DeadOil)
//   {
//     // B_mass = c2o/c1w * c1p + c2p;
//     // J = +c2o/c1w*well_Js[0] + well_Js[1];
//     entry += (c2o/c1w * c1p + c2p)/time_step;
//     entry += +c2o/c1w*well_Js[0] + well_Js[1];

//   }
//   else if (model.fluid_model == Model::FluidModelType::Blackoil)
//   {
//     const double A = c2o/c1w * (c3g-c3w)/(c3g-c3o);
//     const double B = c2o / (c3g - c3o);
//     // B_mass = A*c1p + c2p + B*c3p;
//     entry += (A*c1p + c2p + B*c3p)/time_step;

//     // need to get equations for J
//     AssertThrow(false, ExcNotImplemented());
//   }

//   // coupling with geomechanics
//   if (model.coupling_strategy() == Model::FluidCouplingStrategy::FixedStressSplit)
//   {
//     const double alpha = model.get_biot_coefficient();
//     const double E = model.get_young_modulus->value(this->cell_coord);
//     const double nu = model.get_poisson_ratio->value(this->cell_coord);
//     const double bulk_modulus = E/3.0/(1.0-2.0*nu);
//     entry += cell_volume/B_w * alpha*alpha/bulk_modulus/time_step;
//   }

//   return entry;
// } // eom




// template <int dim>
// double
// IMPESPressure<2, dim>::
// get_rhs_cell_entry(const double time_step,
//                    const double pressure,
//                    const double,
//                    const int) const
// {
//   double entry = 0;

//   const double c1w = this->saturation_terms[0][0];
//   const double c2o = this->saturation_terms[1][1];
//   const double c1p = this->pressure_terms[0];
//   const double c2p = this->pressure_terms[1];

//   if (this->model.fluid_model == Model::FluidModelType::DeadOil)
//   {
//     entry += (c2o/c1w * c1p + c2p)*pressure/time_step;  // B matrix
//     entry += +c2o/c1w * this->well_Qs[0] + this->well_Qs[1]; // Q vector
//     // entry += - (c2o/c1w*c1e + c2e)*delta_div_u/time_step; // poroelastic
//     // entry += - (c2o/c1w*c1e + c2e)*delta_div_u/time_step; // poroelastic
//   }
//   else if (this->model.fluid_model == Model::FluidModelType::WaterGas)
//   {
//     AssertThrow(false, ExcNotImplemented());
//   }

//   return entry;
// }  // eom



// template <int dim>
// double
// IMPESPressure<3, dim>::
// get_rhs_cell_entry(const double time_step,
//                    const double pressure,
//                    const double,
//                    const int) const
// {
//   AssertThrow(false, ExcNotImplemented());

//   double entry = 0;

//   const double c1w = this->saturation_terms[0][0];
//   const double c2o = this->saturation_terms[1][1];
//   const double c3w = this->saturation_terms[2][0];
//   const double c3o = this->saturation_terms[2][1];
//   const double c3g = this->saturation_terms[2][2];
//   const double c1p = this->pressure_terms[0];
//   const double c2p = this->pressure_terms[1];
//   const double c3p = this->pressure_terms[2];

//   const double A = c2o/c1w * (c3g-c3w)/(c3g-c3o);
//   const double B = c2o / (c3g - c3o);
//   entry += (A*c1p + c2p + B*c3p)*pressure/time_step;  // B matrix
//   return entry;
// }  // eom


// template <int n_phases>
// double
// IMPESPressure<dim,n_phases>::
// get_rhs_cell_entry(const double time_step,
//                    const double pressure,
//                    const double,
//                    const int) const
// {
//   // two last variables are ignored (used in children)

//   double entry = 0;
//   const auto & model = this->model;
//   if (model.fluid_model == Model::FluidModelType::Liquid)
//   {
//     entry += c1p * pressure/time_step; // B matrix
//     entry += well_Qs[0];  // Q vector
//     entry += -c1e*delta_div_u/time_step; // poroelastic
//     // std::cout << -c1e*delta_div_u/time_step << std::endl;
//   }
//   else if (model.fluid_model == Model::FluidModelType::DeadOil)
//   {
//     entry += (c2o/c1w * c1p + c2p)*pressure/time_step;  // B matrix
//     entry += +c2o/c1w*well_Qs[0] + well_Qs[1]; // Q vector
//     entry += - (c2o/c1w*c1e + c2e)*delta_div_u/time_step; // poroelastic
//   }
//   else if (model.fluid_model == Model::FluidModelType::Blackoil)
//   {
//     const double A = c2o/c1w * (c3g-c3w)/(c3g-c3o);
//     const double B = c2o / (c3g - c3o);
//     entry += (A*c1p + c2p + B*c3p)*pressure/time_step;  // B matrix

//     // need to get equations for J and Q and poroelastic
//     AssertThrow(false, ExcNotImplemented());
//   }

//   // coupling with geomechanics
//   if (model.coupling_strategy() == Model::FluidCouplingStrategy::FixedStressSplit)
//   {
//     const double alpha = model.get_biot_coefficient();
//     const double E = model.get_young_modulus->value(this->cell_coord);
//     const double nu = model.get_poisson_ratio->value(this->cell_coord);
//     const double bulk_modulus = E/3.0/(1.0-2.0*nu);
//     entry += alpha*alpha/bulk_modulus/time_step * pressure;
//     // std::cout << alpha*alpha/bulk_modulus/time_step * pressure << std::endl;
//   }
//   return entry;
// } // eom



// template <int dim>
// inline
// double
// IMPESPressure<2,dim>::get_matrix_face_entry(const int) const
// {
//   double entry = 0;

//   if (this->model.fluid_model == Model::FluidModelType::DeadOil)
//   {
//     const double c2o = this->saturation_terms[1][1];
//     const double c1w = this->saturation_terms[0][0];
//     const double c1p = this->pressure_terms[0];
//     entry += +c2o/c1w * this->face_transmissibility[0] + this->face_transmissibility[1];
//   }
//   else
//   {
//     AssertThrow(false, ExcNotImplemented());
//   }

//   return entry;
// } // eom


// template <int dim>
// inline
// double
// IMPESPressure<3,dim>::get_matrix_face_entry(const int) const
// {
//   AssertThrow(false, ExcNotImplemented());
//   return 0;
// } // eom



// template <int dim>
// inline
// double
// IMPESPressure<2, dim>::get_rhs_face_entry(const double,
//                                          const int) const
// {
//   double entry = 0;

//   if (this->model.fluid_model == Model::FluidModelType::DeadOil)
//   {
//     const double c2o = this->saturation_terms[1][1];
//     const double c1w = this->saturation_terms[0][0];
//     entry += +c2o/c1w * this->face_gravity_terms[0]
//                         +
//                         this->face_gravity_terms[1];
//   }
//   else
//   {
//     AssertThrow(false, ExcNotImplemented());
//   }

//   return entry;
// } // eom



// template <int dim>
// inline
// double
// IMPESPressure<3, dim>::get_rhs_face_entry(const double,
//                                           const int) const
// {
//   AssertThrow(false, ExcNotImplemented());
//   return 0;
// } // eom

} // end namespace

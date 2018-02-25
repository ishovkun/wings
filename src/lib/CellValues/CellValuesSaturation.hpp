#pragma once

#include <CellValues/CellValuesPressure.hpp>


namespace CellValues
{
using namespace dealii;

/*
 * This class is a modification of CellValuesPressure and provides
 * methods to update rhs vector for the
 * IMPES saturation solver
 */
template <int dim>
class CellValuesSaturation : public CellValuesPressure<dim>
{
 public:
  CellValuesSaturation(const Model::Model<dim> &model);
  /* Update storage vectors and values for the current face */
  virtual void update_face_values(const CellValuesPressure<dim> & neighbor_data,
                                  const FaceGeometry            & face_values);
  /* Get a rhs entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_rhs_cell_entry(const double time_step,
                                    const double pressure,
                                    const double old_pressure,
                                    const int    phase) const;
  /* Get a rhs entry corresponding to the face.
   * should be called once per face after update_face_values()
   */
  virtual double get_rhs_face_entry(const double time_step,
                                    const int    phase) const;
  // Variables
 protected:
  double pressure_difference;
};



template <int dim>
CellValuesSaturation<dim>::CellValuesSaturation(const Model::Model<dim> &model_)
    :
    CellValuesPressure<dim>::CellValuesPressure(model_)
{}



template<int dim>
void
CellValuesSaturation<dim>::
update_face_values(const CellValuesPressure<dim> & neighbor_data,
                   const FaceGeometry            & face_values)
{
  CellValuesPressure<dim>::update_face_values(neighbor_data, face_values);
  pressure_difference = this->pressure - neighbor_data.pressure;
}  // end update_face_values



template<int dim>
inline
double
CellValuesSaturation<dim>::get_rhs_face_entry(const double time_step,
                                              const int phase) const
{
  double result = 0;
  if (CellValuesPressure<dim>::model.fluid_model == Model::FluidModelType::Liquid)
  {
    AssertThrow(false, ExcMessage("Cannot solve for single phase"));
  }
  else if (CellValuesPressure<dim>::model.fluid_model == Model::FluidModelType::DeadOil)
  {
    if (phase == 0)
    {
      result += - this->T_w_face * pressure_difference / this->c1w;
      result += this->G_w_face / this->c1w;
    }
    else
    {
      result += - this->T_o_face * pressure_difference / this->c2o;
      result += this->G_o_face / this->c2o;
    }
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return time_step*result;
} // eom



template<int dim>
inline
double
CellValuesSaturation<dim>::get_rhs_cell_entry(const double time_step,
                                              const double pressure,
                                              const double old_pressure,
                                              const int phase) const
{
  double result = 0;
  if (this->model.fluid_model == Model::FluidModelType::Liquid)
  {
    AssertThrow(false, ExcMessage("Cannot solve for single phase"));
  }
  else if (this->model.fluid_model == Model::FluidModelType::DeadOil)
  {
    if (phase == 0)
    {
      result += this->vector_Q_phase[0] / this->c1w *
          time_step;
      result += this->c1p / this->c1w *
      (pressure - old_pressure);
      result += -this->c1e * this->delta_div_u;
    }
    else
    {
      result += this->vector_Q_phase[1] / this->c2o *
          time_step;
      result += this->c2p / this->c2o *
          (pressure - old_pressure);
      result += -this->c2e * this->delta_div_u;
    }
  }
  else
    AssertThrow(false, ExcNotImplemented());


  return result;
}  // eom

}  // end of namespace

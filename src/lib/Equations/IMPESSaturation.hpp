#pragma once

#include <Equations/IMPESPressure.hpp>


namespace Equations
{
using namespace dealii;

/*
 * This class is a modification of IMPESPressure and provides
 * methods to update rhs vector for the
 * IMPES saturation solver
 */
template <int n_phases,int dim>
class IMPESSaturation : public IMPESPressure<n_phases,dim>
{
 public:
  IMPESSaturation(const Model::Model<dim> &model);
  /* Update storage vectors and values for the current face */
  virtual void update_face_values(const CellIterator<dim> & neighbor_cell,
                                  const SolutionValues    & solution_values,
                                  const FaceGeometry      & geometry) override;
  /* Get a rhs entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_rhs_cell_entry(const double time_step,
                                    const double pressure,
                                    const double old_pressure,
                                    const int    phase) const override;
  /* Get a rhs entry corresponding to the face.
   * should be called once per face after update_face_values()
   */
  virtual double get_rhs_face_entry(const double time_step,
                                    const int    phase) const override;
  // Variables
 protected:
  double pressure_difference;
};



template <int n_phases,int dim>
IMPESSaturation<n_phases,dim>::IMPESSaturation(const Model::Model<dim> &model_)
    :
    IMPESPressure<dim>::IMPESPressure(model_)
{}



template <int n_phases,int dim>
void
IMPESSaturation<n_phases,dim>::
update_face_values(const CellIterator<dim> & neighbor_cell,
                   const SolutionValues    & neighbor_solution,
                   const FaceGeometry      & geometry)
{
  IMPESPressure<n_phases,dim>::update_face_values(neighbor_cell,
                                                  neighbor_solution,
                                                  geometry);
  pressure_difference = this->pressure - neighbor_solution.pressure;
}  // end update_face_values



template <int n_phases,int dim>
inline
double
IMPESSaturation<n_phases,dim>::get_rhs_face_entry(const double time_step,
                                                  const int phase) const
{
  double entry = 0;

  entry += -this->T_face[phase] * pressure_difference /
      this->saturation_terms[phase,phase];
  entry += this->G_face[phase] / this->saturation_terms[phase,phase];

  if (n_phases == 3)
      AssertThrow(false, ExcNotImplemented());

  return entry;
} // eom



template <int n_phases,int dim>
inline
double
IMPESSaturation<n_phases,dim>::
get_rhs_cell_entry(const double time_step,
                   const double pressure,
                   const double old_pressure,
                   const int phase) const
{
  double entry = 0;

  entry += this->well_Qs[phase] * time_step / this->saturation_terms[phase,phase];
  entry += this->pressure_terms[phase] / this->saturation_terms[phase,phase] *
      (pressure - old_pressure);

  // entry += -this->c1e * this->delta_div_u;

  return entry;
}  // eom

}  // end of namespace

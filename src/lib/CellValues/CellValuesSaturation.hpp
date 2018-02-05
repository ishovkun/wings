#pragma once

#include <CellValues/CellValuesBase.hpp>


namespace CellValues
{
using namespace dealii;


/*
 * This class is a modification of CellValuesBase and provides
 * methods to update rhs vector for the
 * IMPES saturation solver
 */
template <int dim>
class CellValuesSaturation : public CellValuesBase<dim>
{
 public:
  CellValuesSaturation(const Model::Model<dim> &model);
  /* Update storage vectors and values for the current face */
  virtual void update_face_values(const CellValuesBase<dim> &neighbor_data,
                                  const Tensor<1,dim>       &face_normal,
                                  const double               dS);
  /* Get a rhs entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_rhs_cell_entry(const double time_step,
                                    const double pressure,
                                    const double old_pressure,
                                    const int phase) const;
  /* Get a rhs entry corresponding to the face.
   * should be called once per face after update_face_values()
   */
  virtual double get_rhs_face_entry(const double time_step,
                                    const int phase) const;
  // Variables
 private:
  double pressure_difference;
};



template <int dim>
CellValuesSaturation<dim>::CellValuesSaturation(const Model::Model<dim> &model_)
    :
    CellValuesBase<dim>::CellValuesBase(model_)
{}



template<int dim>
void
CellValuesSaturation<dim>::update_face_values(const CellValuesBase<dim> &neighbor_data,
                                              const Tensor<1,dim>       &face_normal,
                                              const double               dS)
{
  CellValuesBase<dim>::update_face_values(neighbor_data, face_normal, dS);
  pressure_difference = CellValuesBase<dim>::pressure - neighbor_data.pressure;
}  // end update_face_values


// template<int dim>
// inline
// double
// CellValuesSaturation<dim>::get_B(const int phase) const
// {
//   double result = 0;
//   if (CellValuesBase<dim>::model.type == Model::ModelType::SingleLiquid)
//   {
//     AssertThrow(false, ExcMessage("Cannot solve for single phase"));
//   }
//   else if (CellValuesBase<dim>::model.type == Model::ModelType::WaterOil)
//   {
//     if (phase == 0)
//     {
//       // std::cout << "CellValuesBase<dim>::c1p = "<< CellValuesBase<dim>::c1p << std::endl;
//       // std::cout << "CellValuesBase<dim>::c1w = "<< CellValuesBase<dim>::c1w << std::endl;
//       result = CellValuesBase<dim>::c1p / CellValuesBase<dim>::c1w;
//     }
//     else
//     {
//       result = CellValuesBase<dim>::c2p / CellValuesBase<dim>::c2o;
//     }
//   }
//   else
//     AssertThrow(false, ExcNotImplemented());

//   return result;
// }  // end get_B



// template<int dim>
// inline
// double
// CellValuesSaturation<dim>::get_Q(const int phase) const
// {
//   double result = 0;
//   if (this->model.type == Model::ModelType::SingleLiquid)
//   {
//     AssertThrow(false, ExcMessage("Cannot solve for single phase"));
//   }
//   else if (this->model.type == Model::ModelType::WaterOil)
//   {
//     if (phase == 0)
//       result = CellValuesBase<dim>::vector_Q_phase[0] / CellValuesBase<dim>::c1w;
//     else
//       result = CellValuesBase<dim>::vector_Q_phase[1] / CellValuesBase<dim>::c2o;
//   }
//   else
//     AssertThrow(false, ExcNotImplemented());

//   return result;
// }  // end get_Q




// template<int dim>
// inline
// double
// CellValuesSaturation<dim>::get_T_face(const int phase) const
// {
//   double result = 0;
//   if (CellValuesBase<dim>::model.type == Model::ModelType::SingleLiquid)
//   {
//     AssertThrow(false, ExcMessage("Cannot solve for single phase"));
//   }
//   else if (CellValuesBase<dim>::model.type == Model::ModelType::WaterOil)
//   {
//     if (phase == 0)
//       result = - CellValuesBase<dim>::T_w_face * pressure_difference / CellValuesBase<dim>::c1w;
//     else
//       result = - CellValuesBase<dim>::T_o_face * pressure_difference / CellValuesBase<dim>::c2o;
//   }
//   else
//     AssertThrow(false, ExcNotImplemented());

//   return result;
// }  // end get_T_face



// template<int dim>
// inline
// double
// CellValuesSaturation<dim>::get_G_face(const int phase) const
// {
//   double result = 0;
//   if (CellValuesBase<dim>::model.type == Model::ModelType::SingleLiquid)
//   {
//     AssertThrow(false, ExcMessage("Cannot solve for single phase"));
//   }
//   else if (CellValuesBase<dim>::model.type == Model::ModelType::WaterOil)
//   {
//     if (phase == 0)
//       result = CellValuesBase<dim>::G_w_face / CellValuesBase<dim>::c1w;
//     else
//       result = CellValuesBase<dim>::T_o_face / CellValuesBase<dim>::c2o;
//   }
//   else
//     AssertThrow(false, ExcNotImplemented());

//   return result;
// }  // end get_B_face



template<int dim>
inline
double
CellValuesSaturation<dim>::get_rhs_face_entry(const double time_step,
                                              const int phase) const
{
  double result = 0;
  if (CellValuesBase<dim>::model.type == Model::ModelType::SingleLiquid)
  {
    AssertThrow(false, ExcMessage("Cannot solve for single phase"));
  }
  else if (CellValuesBase<dim>::model.type == Model::ModelType::WaterOil)
  {
    if (phase == 0)
    {
      result += - CellValuesBase<dim>::T_w_face * pressure_difference / CellValuesBase<dim>::c1w;
      result += CellValuesBase<dim>::G_w_face / CellValuesBase<dim>::c1w;
    }
    else
    {
      result += - CellValuesBase<dim>::T_o_face * pressure_difference / CellValuesBase<dim>::c2o;
      result += CellValuesBase<dim>::G_o_face / CellValuesBase<dim>::c2o;
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
  if (CellValuesBase<dim>::model.type == Model::ModelType::SingleLiquid)
  {
    AssertThrow(false, ExcMessage("Cannot solve for single phase"));
  }
  else if (CellValuesBase<dim>::model.type == Model::ModelType::WaterOil)
  {
    if (phase == 0)
    {
      result += CellValuesBase<dim>::vector_Q_phase[0] / CellValuesBase<dim>::c1w *
          time_step;
      result += CellValuesBase<dim>::c1p / CellValuesBase<dim>::c1w *
      (pressure - old_pressure);
    }
    else
    {
      result += CellValuesBase<dim>::vector_Q_phase[1] / CellValuesBase<dim>::c2o *
          time_step;
      result += CellValuesBase<dim>::c2p / CellValuesBase<dim>::c2o *
          (pressure - old_pressure);
    }
  }
  else
    AssertThrow(false, ExcNotImplemented());


  return result;
}  // eom

}  // end of namespace

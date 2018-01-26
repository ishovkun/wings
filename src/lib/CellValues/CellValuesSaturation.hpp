#pragma once

#include <CellValues/CellValuesBase.hpp>


namespace CellValues
{
using namespace dealii;


template <int dim>
class CellValuesSaturation : public CellValuesBase<dim>
{
 public:
  CellValuesSaturation(const Model::Model<dim> &model);
  virtual void update_face_values(const CellValuesBase<dim> &neighbor_data,
                                  const Tensor<1,dim>       &face_normal,
                                  const double               dS);
  double get_T_face(const int phase) const;
  double get_G_face(const int phase) const;
  double get_B(const int phase) const;
  double get_Q(const int phase) const;
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
  pressure_difference = this->pressure - neighbor_data.pressure;
}  // end update_face_values


template<int dim>
inline
double
CellValuesSaturation<dim>::get_B(const int phase) const
{
  double result = 0;
  if (this->model.type == Model::ModelType::SingleLiquid)
  {
    AssertThrow(false, ExcMessage("Cannot solve for single phase"));
  }
  else if (this->model.type == Model::ModelType::WaterOil)
  {
    if (phase == 0)
      result = this->c1p / this->c1w;
    else
      result = this->c2p / this->c1p;
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return result;
}  // end get_T_face



template<int dim>
inline
double
CellValuesSaturation<dim>::get_Q(const int phase) const
{
  double result = 0;
  if (this->model.type == Model::ModelType::SingleLiquid)
  {
    AssertThrow(false, ExcMessage("Cannot solve for single phase"));
  }
  else if (this->model.type == Model::ModelType::WaterOil)
  {
    if (phase == 0)
      result = this->vector_Q_phase[0] / this->c1w;
    else
      result = this->vector_Q_phase[1] / this->c1p;
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return result;
}  // end get_T_face




template<int dim>
inline
double
CellValuesSaturation<dim>::get_T_face(const int phase) const
{
  double result = 0;
  if (this->model.type == Model::ModelType::SingleLiquid)
  {
    AssertThrow(false, ExcMessage("Cannot solve for single phase"));
  }
  else if (this->model.type == Model::ModelType::WaterOil)
  {
    if (phase == 0)
      result = this->T_w_face * pressure_difference / this->c1w;
    else
      result = this->T_o_face * pressure_difference / this->c2o;
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return result;
}  // end get_T_face



template<int dim>
inline
double
CellValuesSaturation<dim>::get_G_face(const int phase) const
{
  double result = 0;
  if (this->model.type == Model::ModelType::SingleLiquid)
  {
    AssertThrow(false, ExcMessage("Cannot solve for single phase"));
  }
  else if (this->model.type == Model::ModelType::WaterOil)
  {
    if (phase == 0)
      result = this->G_w_face / this->c1w;
    else
      result = this->T_o_face / this->c2o;
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return result;
}  // end get_T_face

}  // end of namespace

#pragma once

#include <FEFunction/FEFunction.hpp>

namespace FEFunction
{
using namespace dealii;

template <int dim, typename VectorType>
class FEFunctionPS : public FEFunction<dim, VectorType>
{
 public:
  FEFunctionPS(const DoFHandler<dim>         & dof_handler,
               const VectorType              & pressure,
               const std::vector<VectorType> & saturation);

  void vector_value(const Point<dim> & p,
                    Vector<double>   & dst) override;

  const VectorType & pressure;

};



template <int dim, typename VectorType>
FEFunctionPS<dim,VectorType>::FEFunctionPS(const DoFHandler<dim>         & dof_handler,
                                           const VectorType              & pressure,
                                           const std::vector<VectorType> & saturation)
    :
    FEFunction<dim, VectorType>(dof_handler, saturation),
    pressure(pressure)
{}  // end FEFunctionPS



template <int dim, typename VectorType>
void
FEFunctionPS<dim,VectorType>::vector_value(const Point<dim> & p,
                                           Vector<double>   & dst)
{
  //                           saturation + pressure
  AssertThrow(dst.size() == this->vectors.size() + 1,
              ExcDimensionMismatch(dst.size(), this->vectors.size()+1));

  const auto & fe = this->dof_handler.get_fe();

  QGauss<dim>   quadrature_formula(1);
  FEValues<dim> fe_values(fe, quadrature_formula, update_values);

  std::vector<double>  v_values(quadrature_formula.size());

  // set vector to zero
  for (auto & value: dst)
    value = 0;

  typename DoFHandler<dim>::active_cell_iterator
      cell = this->dof_handler.begin_active(),
		  endc = this->dof_handler.end();

  for (; cell != endc; ++cell)
    if (!cell->is_artificial())
      if (cell->point_inside(p))
      {
        fe_values.reinit(cell);

        fe_values.get_function_values(pressure, v_values);
        dst[0] = v_values[0];

        for (unsigned int c=0; c<this->vectors.size(); ++c)
        {
          fe_values.get_function_values(this->vectors[c], v_values);
          dst[c+1] = v_values[0];
        }
        break;
      }  // end cell loop

}  // end vector_value

} // end namespace

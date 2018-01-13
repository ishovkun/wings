#pragma once

#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>



namespace FEFunction
{
using namespace dealii;


template <int dim, typename VectorType>
class FEFunction : public Function<dim>
{
 public:
  FEFunction(const DoFHandler<dim>         &dof_handler,
             const std::vector<VectorType> &vectors);
  void vector_value(const Point<dim>    &p,
                    Vector<double> &dst) const;
 private:
  const DoFHandler<dim>          &dof_handler;
  const std::vector<VectorType> vectors;
};



template <int dim, typename VectorType>
FEFunction<dim, VectorType>::
FEFunction(const DoFHandler<dim>         &dof_handler_,
           const std::vector<VectorType> &vectors_)
    :
    dof_handler(dof_handler_),
    vectors(vectors_)
{}  // eom



template <int dim, typename VectorType>
void
FEFunction<dim,VectorType>::vector_value(const Point<dim> &p,
                                         Vector<double>   &dst) const
{
  /* Don't call this function before setup_dofs */
  AssertThrow(dof_handler.has_active_dofs(), ExcMessage("DofHandler is empty"));

  const auto & fe = dof_handler.get_fe();

  QGauss<dim>   quadrature_formula(1);
  FEValues<dim> fe_values(fe, quadrature_formula, update_values);

  std::vector<double>  v_values(quadrature_formula.size());

  // set vector to zero
  for (auto & value: dst)
    value = 0;

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

  for (; cell != endc; ++cell)
    if (!cell->is_artificial())
      if (cell->point_inside(p))
      {
        fe_values.reinit(cell);
        for (unsigned int c=0; c<vectors.size(); ++c)
        {
          fe_values.get_function_values(vectors[c], v_values);
          dst[c] = v_values[0];
        }
        break;
      }  // end cell loop

}  // eom


}  // end of namespace

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
  FEFunction(const DoFHandler<dim>         & dof_handler,
             const std::vector<VectorType> & vectors);
  FEFunction(const DoFHandler<dim>         & dof_handler_,
             const VectorType              & vector);
  virtual void vector_value(const Point<dim> & p,
                            Vector<double>   & dst) const;
  virtual double value(const Point<dim> & p,
                       const unsigned int component=0) const;

 protected:
  const DoFHandler<dim>          & dof_handler;
  const std::vector<VectorType>  & vectors;
  const VectorType                 dummy; // to supress compilation warning
  const std::vector<VectorType>    dummy_std;
  const VectorType               & single_vector;
};



template <int dim, typename VectorType>
FEFunction<dim, VectorType>::
FEFunction(const DoFHandler<dim>         & dof_handler,
           const std::vector<VectorType> & vectors_)
    :
    dof_handler(dof_handler),
    vectors(vectors_),
    single_vector(dummy)
{}  // eom



template <int dim, typename VectorType>
FEFunction<dim, VectorType>::
FEFunction(const DoFHandler<dim>         & dof_handler,
           const VectorType              & vector)
    :
    dof_handler(dof_handler),
    vectors(dummy_std),
    single_vector(vector)
{}  // eom



template <int dim, typename VectorType>
double
FEFunction<dim,VectorType>::value(const Point<dim> & p,
                                  const unsigned int component) const
{
  /* Don't call this function before setup_dofs */
  AssertThrow(dof_handler.has_active_dofs(), ExcMessage("DofHandler is empty"));
  // AssertThrow(vectors.size() == 0, ExcMessage("Either vectors or single_vector should be empty"));
  // AssertThrow(component == 0, ExcNotImplemented());

  const auto & fe = dof_handler.get_fe();

  QGauss<dim>   quadrature_formula(1);
  FEValues<dim> fe_values(fe, quadrature_formula, update_values);
  std::vector<double>  v_values(quadrature_formula.size());

  double result = 0;

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

  for (; cell != endc; ++cell)
    if (!cell->is_artificial())
      if (cell->point_inside(p))
      {
        fe_values.reinit(cell);
        if (vectors.size() == 0)
          fe_values.get_function_values(single_vector, v_values);
        else
          fe_values.get_function_values(vectors[component], v_values);
        result = v_values[0];
        break;
      }  // end cell loop

  return result;
}



template <int dim, typename VectorType>
void
FEFunction<dim,VectorType>::vector_value(const Point<dim> & p,
                                         Vector<double>   & dst) const
{
  /* Don't call this function before setup_dofs */
  AssertThrow(dof_handler.has_active_dofs(), ExcMessage("DofHandler is empty"));

  AssertThrow(dst.size() == vectors.size()+1,
              ExcDimensionMismatch(dst.size(), vectors.size()+1));

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
        fe_values.get_function_values(this->single_vector, v_values);
        dst[0] = v_values[0];

        for (unsigned int c=0; c<vectors.size(); ++c)
        {
          fe_values.get_function_values(vectors[c], v_values);
          dst[c+1] = v_values[0];
        }
        break;
      }  // end cell loop

}  // eom


}  // end of namespace

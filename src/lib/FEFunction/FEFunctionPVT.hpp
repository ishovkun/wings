#pragma once

#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <LookupTable.hpp>


namespace FEFunction
{
using namespace dealii;


template <int dim, typename VectorType>
class FEFunctionPVT : public Function<dim>
{
 public:
  FEFunctionPVT(const DoFHandler<dim>            &dof_handler,
                const VectorType                 &vector,
                const Interpolation::LookupTable &pvt_table);
  void vector_value(const Point<dim> &p,
                    Vector<double>   &dst) const;
 private:
  const DoFHandler<dim>            &dof_handler;
  const VectorType                 &vector;
  const Interpolation::LookupTable &pvt_table;
};



template <int dim, typename VectorType>
FEFunctionPVT<dim, VectorType>::
FEFunctionPVT(const DoFHandler<dim>            &dof_handler_,
              const VectorType                 &vector,
              const Interpolation::LookupTable &pvt_table)
    :
    dof_handler(dof_handler_),
    vector(vector),
    pvt_table(pvt_table)
{}  // eom



template <int dim, typename VectorType>
void
FEFunctionPVT<dim,VectorType>::vector_value(const Point<dim> &p,
                                            Vector<double>   &dst) const
{
  /* Don't call this function before setup_dofs */
  AssertThrow(dof_handler.has_active_dofs(), ExcMessage("DofHandler is empty"));

  const auto & fe = dof_handler.get_fe();

  QGauss<dim>         quadrature_formula(1);
  FEValues<dim>       fe_values(fe, quadrature_formula, update_values);
  std::vector<double> lut_values(dst.size());
  std::vector<double> v_values(quadrature_formula.size());

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
        fe_values.get_function_values(vector, v_values);
        pvt_table.get_values(v_values[0], lut_values);
        for (unsigned int i=0; i<dst.size(); ++i)
          dst[i] = lut_values[i];

        break;
      }  // end cell loop

}  // eom


}  // end of namespace

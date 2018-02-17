#pragma once

#include <deal.II/numerics/data_postprocessor.h>


namespace Output
{

template <int dim>
class ScaleOutputVector : public DataPostprocessorScalar<dim>
{
 public:
  ScaleOutputVector(const std::string variable_name,
                    const double      factor);
  virtual void
  compute_derived_quantities_vector(const std::vector<Vector<double> >               & uh,
                                    const std::vector<std::vector<Tensor<1, dim> > > & duh,
                                    const std::vector<std::vector<Tensor<2, dim> > > & dduh,
                                    const std::vector<Point<dim> >                   & normals,
                                    const std::vector<Point<dim> >                   & evaluation_points,
                                    std::vector<Vector<double> >                     & computed_quantities) const;
  const double factor;
};

template<int dim>
ScaleOutputVector<dim>::ScaleOutputVector(const std::string variable_name,
                                          const double      factor)
    :
    DataPostprocessorScalar<dim>(variable_name, update_values),
    factor(factor)
{}  // end ScaleOutputVector



template<int dim>
void
ScaleOutputVector<dim>::
compute_derived_quantities_vector(const std::vector<Vector<double> >               & uh,
                                  const std::vector<std::vector<Tensor<1, dim> > > & duh,
                                  const std::vector<std::vector<Tensor<2, dim> > > & dduh,
                                  const std::vector<Point<dim> >                   & normals,
                                  const std::vector<Point<dim> >                   & evaluation_points,
                                  std::vector<Vector<double> >                     & computed_quantities) const
{
  Assert(computed_quantities.size() == uh.size(),
         ExcDimensionMismatch (computed_quantities.size(), uh.size()));

  for (unsigned int i=0; i<computed_quantities.size(); i++)
  {
    Assert(computed_quantities[i].size() == 1,
           ExcDimensionMismatch (computed_quantities[i].size(), 1));
    Assert(uh[i].size() == 1, ExcDimensionMismatch (uh[i].size(), 1));
    computed_quantities[i](0) = uh[i](0)*factor;
  }
}  // end   compute_derived_quantities_vector

} // end of namespace

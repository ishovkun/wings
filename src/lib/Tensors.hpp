#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>


namespace Tensors
{
  using namespace dealii;


  template <int dim>
  inline Tensor<1,dim>
  get_unit_vector()
  {
    Tensor<1,dim> unit_vector;
    for (int c=0; c<dim; c++)
      unit_vector[c] = 1;
    return unit_vector;
  }  // eof

  template <int dim>
  inline Tensor<2,dim>
  get_identity_tensor()
  {
    Tensor<2,dim> identity_tensor;
    identity_tensor.clear();
    for (int i=0; i<dim; ++i)
      identity_tensor[i][i] = 1;
    return identity_tensor;
  }  // eof

}

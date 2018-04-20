#pragma once

#include <deal.II/base/tensor_function.h>

#include <BitMap/BitMapFunction.hpp>

namespace BitMap {

using namespace dealii;


template <int rank, int dim>
class BitMapTensorFunction : public TensorFunction<rank, dim, double>
{
 public:
  BitMapTensorFunction(const std::string      & filename,
                       const Tensor<rank,dim> & anisotropy);

  Tensor<rank,dim> value(const Point<dim> & p) const;
  void scale_coordinates(const double scale);

 private:
  BitMapFunction<dim>  func;
  const Tensor<1, dim> anisotropy;
};



template <int rank, int dim>
BitMapTensorFunction<rank,dim>::
BitMapTensorFunction(const std::string      & filename,
                     const Tensor<rank,dim> & anisotropy)
    :
    func(filename),
    anisotropy(anisotropy)
{}



template <int rank, int dim>
Tensor<rank, dim>
BitMapTensorFunction<rank,dim>::value(const Point<dim> & p) const
{
  return anisotropy*func.value(p);
}  // end value



template <int rank, int dim>
void
BitMapTensorFunction<rank,dim>::scale_coordinates(const double scale)
{
  func.scale_coordinates(scale);
}  // end value

} // end of namespace

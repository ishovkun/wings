#pragma once

#include <deal.II/base/function.h>

#include <BitMap/BitMapFile.hpp>

namespace Wings
{

namespace BitMap {

using namespace dealii;


template <int dim>
class BitMapFunction : public Function<dim>
{
 public:
  BitMapFunction(const std::string & filename);

  double value(const Point<dim> &p,
               const unsigned int /*component*/ = 0) const;
  // void vector_value(const Point<dim> & p,
  //                   Tensor<1,dim>    & v) const;
  // void vector_value(const Point<dim> & p,
  //                   Vector<double>   & v) const;
  void scale_coordinates(const double scale);

 private:
  // BitMapFile<dim> f;
  BitMapFile f;
};



template<int dim>
BitMapFunction<dim>::BitMapFunction(const std::string & filename)
    :
    Function<dim>(1),
    f(filename)
{}  // eom



template<>
double
BitMapFunction<2>::value(const Point<2> &p,
                         const unsigned int) const
{
  // Assert(c<2, ExcNotImplemented());
  return f.get_value(p(0), p(1));
}  // eom



template<>
double
BitMapFunction<3>::value(const Point<3> &p,
                         const unsigned int) const
{
  // Assert(c<2, ExcNotImplemented());
  return f.get_value(p(0), p(1), p(2));
}  // eom



template<int dim>
void
BitMapFunction<dim>::scale_coordinates(const double scale)
{
  f.scale_coordinates(scale);
}  // eom



}


}  // end Wings

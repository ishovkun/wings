#pragma once

namespace Math
{
using namespace dealii;


void harmonic_mean(const Vector<double> &v1,
                   const Vector<double> &v2,
                   const double         dx1,
                   const double         dx2,
                   Vector<double>       &out)
{
  AssertThrow(v1.size() == v2.size(),
              ExcMessage("Dimension mismatch"));
  AssertThrow(out.size() == v2.size(),
              ExcMessage("Dimension mismatch"));
  for (unsigned int i=0; i<v1.size(); ++i){
    if (v1[i] == 0.0 || v2[i] == 0.0)
      out[i] = 0;
    else
      out[i] = (dx1 + dx2)/(dx1/v1[i] + dx2/v2[i]);
  }
}  // eom



void harmonic_mean(const Vector<double> &v1,
                   const Vector<double> &v2,
                   Vector<double>       &out)
{
  harmonic_mean(v1, v2, 1.0, 1.0, out);
}  // eom



double arithmetic_mean(const double x1,
                       const double x2)
{
  return 0.5*(x1+x2);
}  // eom



inline
double upwind(const double x1,
              const double x2,
              const double potential1,
              const double potential2)
{
  if (potential1 >= potential2)
    return x1;
  else return x2;
}  // eom



double sum(const std::vector<double> & v)
{
  AssertThrow(v.size()> 0, ExcEmptyObject());
  double result = 0;
  for (unsigned int i=0; i<v.size(); i++)
    result += v[i];
  return result;
}  // eom



template <int dim>
Tensor<1,dim> normalize(const Tensor<1,dim> &t)
{
  Tensor<1,dim> normalized = t;
  if (normalized.norm() != 0.0)
    normalized/= normalized.norm();

  return normalized;
}   // eom



double relative_difference(const double numerical,
                           const double analytical)
{
  return abs((numerical - analytical)/ analytical);
}



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

}  // end of namespace

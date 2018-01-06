#pragma once

namespace Math
{
  using namespace dealii;


  void harmonic_mean(const Vector<double> &v1,
                     const Vector<double> &v2,
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
        out[i] = 2./(1./v1[i] + 1./v2[i]);
    }
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
}

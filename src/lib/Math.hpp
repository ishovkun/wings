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
}

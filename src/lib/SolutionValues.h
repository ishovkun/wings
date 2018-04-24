#pragma once

#include <deal.II/base/tensor.h>

namespace Wings
{

  template<int dim, int n_phases>
      struct SolutionValues
      {
        dealii::Tensor<1,n_phases> saturation;
        double pressure;
        dealii::Tensor<2, dim> grad_u, grad_old_u;
      };

}  // end Wings

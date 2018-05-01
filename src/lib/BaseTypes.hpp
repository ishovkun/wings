#pragma once

#include <deal.II/dofs/dof_handler.h>

namespace Wings
{

template <int dim>
using CellIterator = typename DoFHandler<dim>::active_cell_iterator;


}  // end Wings

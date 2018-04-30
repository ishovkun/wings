#pragma once

#include <Wellbore.hpp>

/*
  This class is a simple container for wells
  that allows easy creation and data exchange
  with Wellbore class
*/

namespace Wings
{

namespace Wellbore
{

template<int dim, int n_phases>
class Wells
{
  Wells(MPI_Comm & mpi_communicator);


  MPI_Comm & mpi_communicator;
  std::vector<Wellbore<dim,n_phases> wells;
};


template<int dim, int n_phases>
Wells<dim,n_phases>::Wells(MPI_Comm & mpi_communicator)
    :
    mpi_communicator(mpi_communicator)
{}  // end do_something


template<int dim int n_phases>
void
Wells<dim,n_phases>::add_well(WellInfo<dim> & well_info)
{

}  // end do_something


}  // end wells

} // end wings

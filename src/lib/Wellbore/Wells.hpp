#pragma once

#include <Wellbore/Wellbore.hpp>

/*
  This class is a simple container for wells
  that allows easy creation and data exchange
  with Wellbore class
*/

namespace Wings
{

namespace Wellbore
{

using namespace dealii;

template<int dim, int n_phases>
class Wells
{
 public:
  Wells(const Probe::Probe<dim,n_phases> & probe,
        MPI_Comm                         & mpi_communicator);
  void add_well(const WellInfo<dim> & well_info);

 protected:
  const Probe::Probe<dim,n_phases>  & probe;
  MPI_Comm                          & mpi_communicator;
  std::vector<Wellbore<dim,n_phases>> wells;
};



template<int dim, int n_phases>
Wells<dim,n_phases>::Wells(const Probe::Probe<dim,n_phases> & probe,
                           MPI_Comm                         & mpi_communicator)
    :
    probe(probe),
    mpi_communicator(mpi_communicator)
{}  // end do_something


template<int dim, int n_phases>
void
Wells<dim,n_phases>::add_well(const WellInfo<dim> & well_info)
{
  wells.push_back(
      Wellbore<dim,n_phases>(well_info.locations,
                             well_info.radius,
                             probe,
                             mpi_communicator)
                  );
}  // end do_something


}  // end wells

} // end wings

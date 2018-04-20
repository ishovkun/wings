#pragma once

#include <Math.hpp>

/*
 * Abstract class that defines the interaction with FluidEquation classes
 * This family of classes is used to compute system matrix and ths_vector
 * local entries and essentially contains all the physics of the problem.
 */

namespace Equations
{

using namespace dealii;

static const int dim = 3;
// these three structures should used to pass data to CellValues classes

// reference to the dofhandler's cell object
template <int dim>
using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

// this structure contains pressure, saturation, and displacement at a location
struct SolutionValues
{
  SolutionValues(const int n_phases = 3) {saturation.reinit(n_phases);}
  Vector<double> saturation;
  // double div_u, div_old_u;
  double pressure;
  Tensor<1, 3> grad_u, grad_old_u;
};


// this structure contains face area and outward normal, used to compute
// transmissibility
struct FaceGeometry
{
  double      area;
  Tensor<1,3> normal;
};


// Generic  class that defines the interaction with all fluid solvers in Wings
// template <int dim>
class FluidEquationsBase
{
 public:
  /* Update storage vectors and values for the current cell */
  virtual void update_cell_values(const CellIterator<dim> & cell,
                                  const SolutionValues    & solution_values) = 0;
  /* Update storage vectors and values for the current face */
  virtual void update_face_values(const CellIterator<dim> & neighbor_cell,
                                  const SolutionValues    & solution_values,
                                  const FaceGeometry      & geometry) = 0;
  /* Update wellbore rates and j-indices.
   * The calculated rates are not true rates for
   * pressure-controled wells,
   * Q-vector rather gets the value j_ind*BHP
   */
  virtual void update_wells(const CellIterator<dim> &cell) = 0;
  /* Update wellbore rates.
   * This method gets real rates for both flow- and pressure-
   * controlled wellbores.
   */
  virtual void update_wells(const CellIterator<dim> & cell,
                            const double              pressure) = 0;
  // methods for pressure solver
  /* Get a matrix entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_matrix_cell_entry(const double time_step) const = 0;
  /* Get a rhs entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_rhs_cell_entry(const double time_step,
                                    const double x,
                                    const double old_x,
                                    const int comp = 0) const = 0;
  /* Get a matrix entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_matrix_face_entry(const int comp = 0) const = 0;
  /* Get a rhs entry corresponding to the face.
   * should be called once per face after update_face_values()
   */
  virtual double get_rhs_face_entry(const double time_step,
                                    const int comp = 0) const = 0;
};


}  // end of namespace

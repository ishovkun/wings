#pragma once

#include <Math.hpp>


namespace CellValues
{

using namespace dealii;

// these three structures should used to pass data to CellValues classes

// reference to the dofhandler's cell object
template <int dim>
using CellIterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

// this structure contains pressure, saturation, and displacement at a location
struct SolutionValues
{
  SolutionValues(const int n_phases = 3) {saturation.reinit(n_phases);}
  Vector<double> saturation;
  double div_u, div_old_u;
  double pressure;
};


// this structure contains face area and outward normal, used to compute
// transmissibility
struct FaceGeometry
{
  double      area;
  Tensor<1,3> normal;
};


// Generic  class that defines the interaction with all fluid solvers in Wings
template <int dim>
class CellValuesBase
{
 public:
  // CellValuesBase();
  /* Update storage vectors and values for the current cell */
  virtual void update(const CellIterator<dim> & cell,
                      const SolutionValues    & solution_values) = 0;
  /* Update wellbore rates and j-indices.
   * The calculated rates are not true rates for
   * pressure-controled wells,
   * Q-vector rather gets the value j_ind*BHP
   */
  virtual void update_wells(const CellIterator<dim> &cell) = 0;
  /* Update wellbore rates.
   * This method actually gets real rates for both flow- and pressure-
   * controlled wellbores.
   */
  virtual void update_wells(const CellIterator<dim> & cell,
                            const double              pressure) = 0;
  /* Update storage vectors and values for the current face */
  // virtual void update_face_values(const CellValuesBase<dim> & neighbor_data,
  //                                 const FaceGeometry        & face_values) = 0;
  // methods for pressure solver
  /* Get a matrix entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_matrix_cell_entry(const double time_step) const = 0;
  /* Get a rhs entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_rhs_cell_entry(const double time_step,
                                    const double pressure,
                                    const double old_pressure,
                                    const int /* component */ = 0) const = 0;
  /* Get a matrix entry corresponding to the cell.
   * should be called once after update_values()
   */
  virtual double get_matrix_face_entry() const = 0;
  /* Get a rhs entry corresponding to the face.
   * should be called once per face after update_face_values()
   */
  virtual double get_rhs_face_entry(const double /* time_step */,
                                    const int /* component */ = 0) const = 0;
};



// template <int dim>
// CellValuesBase<dim>::CellValuesBase()
// {}



}  // end of namespace

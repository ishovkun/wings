#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>

#include <Schedule.cc>


namespace Wellbore
{
	using namespace dealii;

	template <int dim>
	class Wellbore : public Function<dim>
  {
  public:
    Wellbore(const std::vector< Point<dim> >& locations_,
             const int                        direction_,
             const double                     radius_);
    void set_control(const Schedule::WellControl& control_);
    Schedule::WellControl get_control();
    void locate(const DoFHandler<dim>& dof_handler);

  private:
    std::vector< Point<dim> > locations;
    int                       direction;
    double                    radius;
    Schedule::WellControl     control;

    std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
  };  // eom


  template <int dim>
  Wellbore<dim>::Wellbore(const std::vector< Point<dim> >& locations_,
                          const int                        direction_,
                          const double                     radius_)
    :
    locations(locations_),
    direction(direction_),
    radius(radius_)
  {
    AssertThrow(direction >= 0 && direction < 3,
                ExcMessage("Wrong well direction"));
    AssertThrow(locations.size() > 0,
                ExcMessage("That ain't no a proper well"));
    AssertThrow(radius > 0,
                ExcMessage("Well radius should be a positive number"));
    // check for duplicates
    for (unsigned int i=1; i<locations.size(); i++)
      AssertThrow((locations[i] - locations[i-1]).norm() > 0,
                  ExcMessage("Duplicates in wellbore locations"));

  } //  eom


  template <int dim>
  inline
  void Wellbore<dim>::set_control(const Schedule::WellControl& control_)
  {
    control = control_;
  }  // eom


  template <int dim>
  inline
  Schedule::WellControl
  Wellbore<dim>::get_control()
  {
    return control;
  }  // eom


  template <int dim>
  void Wellbore<dim>::locate(const DoFHandler<dim>& dof_handler)
  {
    /* Algorithm:
       I. if just one well location, add cell that contains the point.
       If point is on the boundary, assign just one.
       Need to loop through neighbors.

       II. If well segments.
       let the segment be defined with eq x = x0 + at,
       where x is a point on the segment, a is the beginning of the segment,
       a is the vector in the direction of the segment, t is a scalar.
       the cell center location is p0.
       We make three checks:
       1. Calculate the vector d that starts in p and is perpendicular to the
       well segment. If (p+d) is not in the cell, skip the segment.
       2. if (p+d) is in the cell an lies within the segment, add the segment
       3. if (p+d) is in the cell but doesn't lie within the segment,
       check whether the end-points of the segment are in the cell and add the
       segment if needed.
       4. if segment is aligned with the cell face, add it only if the neighbor
       cell doesn't contain it already.
     */

    Point<dim> x0, x1, p;
    Tensor<1,dim> a, d;

    cells.clear();

    typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
      neighbor_cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

	  for (; cell!=endc; ++cell)
    // if (cell->is_locally_owned)
    {
      p = cell->center();

      if(locations.size() == 1)  // case wellbore is one point
      {
        if (cell->point_inside(locations[0]))
        {
          bool contains_neighbour_cell = false;

          // check if neighbors already have this location (on the boundary)
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          {
            unsigned int j = cell->neighbor_index(f);
            unsigned int no_neighbor_index = -1;

            if(j == no_neighbor_index) // if neighbor don't exist
              continue;

            if(cell->neighbor(f)->point_inside(locations[0]))
            {
              // check if neighbor cell is already added
              auto it = std::find(cells.begin(), cells.end(), neighbor_cell);
              if (it != cells.end())
                contains_neighbour_cell = true;
            }
          }

          if (!contains_neighbour_cell)
            cells.push_back(cell);
        }  // end if point inside

      }  // end case I

      else  // well segments
        for (unsigned int i=1; i<locations.size(); i++)
        {
          x0 = locations[i-1];
          x1 = locations[i];
          a = x1 - x0;
          a = a/a.norm();
          // distance from cell center to the line
          d = (x0-p) - a*scalar_product(x0-p, a);

          // if (!cell->point_inside(Point<dim>(p+d)))
          //   continue;
        }
    }
  }  // eom
}  // end of namespace

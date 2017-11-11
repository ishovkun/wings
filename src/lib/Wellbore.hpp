#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <math.h>

#include <DefaultValues.cc>
#include <Schedule.cc>


namespace Wellbore
{
	using namespace dealii;

  template <int dim>
  using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

	template <int dim>
	class Wellbore : public Function<dim>
  {
  public:
    Wellbore(const std::vector< Point<dim> >& locations_,
             const int                        direction_,
             const double                     radius_);
    void set_control(const Schedule::WellControl& control_);
    Schedule::WellControl get_control();
    void locate(const DoFHandler<dim>& dof_handler,
                const FE_DGQ<dim>&     fe);
    const std::vector<CellIterator<dim>> & get_cells();
    const std::vector< Point<dim> >      & get_locations();
  private:
    double get_segment_length(const Point<dim>& start,
                              const CellIterator<dim>& cell,
                              const Tensor<1,dim>& tangent);
    std::vector< Point<dim> > locations;
    int                       direction;
    double                    radius;
    Schedule::WellControl     control;

    std::vector<CellIterator<dim>> cells;
    std::vector<double>            segment_length;
    std::vector< Tensor<1,dim> >   segment_tangent;
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
  inline
  const std::vector<CellIterator<dim>> &
  Wellbore<dim>::get_cells()
  {
    return cells;
  }  // eom


  template <int dim>
  void Wellbore<dim>::locate(const DoFHandler<dim>& dof_handler,
                             const FE_DGQ<dim>&     fe)
  {
    /* Algorithm:
       I. if just one well location, add cell that contains the point.
       And break. so no other cells can claim the well

       II. If well segments.
       let the segment be defined with eq x = x0 + at,
       where x is a point on the segment, x0 is the beginning of the segment,
       a is the vector in the direction of the segment, t is a scalar.
       the cell center location is p0.
       x1 is the end point of the segment.

       point d is the closest point to p0 on the segment.
       vector n is between p0 and d.

       We make three checks:
       0. If cell already added from another well segment -> discard cell
       1. Calculate vectors n and d. If d is not in the cell -> discard segment.
       2. if d is in the cell but lies outside the segment
       and segment end-points are outside too -> discard.
       3. We check whether the wellbore is aligned with the cell face, and if
       yes, assign it to only one cell.
       4. if only touches a cell in a vertex that's also bad

       we also calculate dl - the length of the well segment in each cell
     */

    Point<dim> x0, x1, p0, p1, d, start;
    Tensor<1,dim> a, n, nf;

    // we need fe_face_values to get cell normals
    QGauss<dim-1>     face_quadrature_formula(1);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_normal_vectors);

    cells.clear();

    typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
      // neighbor_cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

	  for (; cell!=endc; ++cell)
    // if (cell->is_locally_owned)
    {
      p0 = cell->center();
      // std::cout << "\nCell " << p0 << std::endl;

      auto it = std::find(cells.begin(), cells.end(), cell);
      if (it != cells.end())
        continue;

      if(locations.size() == 1)  // case wellbore is one point
      {
        // std::cout << "Case one point" << std::endl;
        if (cell->point_inside(locations[0]))
        {
          // std::cout << "Point inside" << std::endl;

          cells.push_back(cell);
          // cell = endc;
          break;
        }  // end if point inside

      }  // end case I

      else  // well segments
        for (unsigned int i=1; i<locations.size(); i++)
        {
          // std::cout << "\nsegment i = " << i << std::endl;
          // std::cout << "\nCell " << p0 << std::endl;

          x0 = locations[i-1];
          x1 = locations[i];
          a = x1 - x0;
          double segment_len = a.norm();
          a = a/segment_len;
          // distance from cell center to the line
          d = x0 + a*scalar_product(p0-x0, a);
          n = d - p0;

          // std::cout << "a " << a << std::endl;
          // std::cout << "d " << d << std::endl;
          // std::cout << "n " << n << std::endl;

          // check d inside cell
          if (!cell->point_inside(d))
          {
            // std::cout << "Too far" << std::endl;
            continue;
          }

          // check if d is between x1 and x0
          // d = x0 + a*td
          double td = scalar_product((p0 - x0), a);
          // x1 = x0 + a*t1
          double t1 = segment_len;

          // std::cout << "td = " << td << std::endl;

          const bool x0_inside = cell->point_inside(x0);
          const bool x1_inside = cell->point_inside(x1);

          if((td < 0 || td > t1) && // distance vector outside segment
             (!x0_inside || !x1_inside)) //end-points
          {
            // std::cout << "d in cell but outside segment" << std::endl;
            continue;
          }

          // initial point to seek segment length
          if (td < 0 && x0_inside)
            start = x0;
          if (td > t1 && x1_inside)
            start = x1;
          if (td >= 0 && td <= t1)
            start = d;

          bool skip_cell = false;
          // check if segment aligned with faces and select the closest cell
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          {
            unsigned int j = cell->neighbor_index(f);
            unsigned int no_neighbor_index = -1;
            if(j != no_neighbor_index) // if this neighbor exists
            {
              fe_face_values.reinit(cell, f);
              nf = fe_face_values.normal_vector(0); // 0 is gauss point
              p1 = cell->neighbor(f)->center();

              const bool face_aligned_with_well =
                abs(scalar_product(n, nf))/n.norm() > cos(DefaultValues::small_angle);

              const bool well_closer_to_neighbour =
                n.norm() >= (p1-d).norm() &&
                j < cell->active_cell_index();

              // if (face_aligned_with_well)
              //   std::cout << "Aligned with face " << j << std::endl;

              // if (well_closer_to_neighbour)
              //   std::cout << "Closer to " << p1 << std::endl;

              if (face_aligned_with_well && well_closer_to_neighbour)
              {
                skip_cell = true;
                break;
              }

            } // end if neighbor exists
          } // end face loop

          // iterate vertices
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
            if ((cell->vertex(v) - d).norm() < DefaultValues::small_number)
              skip_cell = true;


          if (skip_cell)
          {
            // std::cout << "OK but another cell is better" << std::endl;
            continue;
          }

          cells.push_back(cell);
          const double l = get_segment_length(start, cell, a);
          segment_length.push_back(l);
          segment_tangent.push_back(a);
        } // end loop segments

    }  // end cell loop
  }  // eom


  template <int dim>
  double Wellbore<dim>::
  get_segment_length(const Point<dim>& start,
                     const CellIterator<dim>& cell,
                     const Tensor<1,dim>& tangent)
  {
    /* Assuming that the start point is in the cell,
       calculate the length of the well segment in the cell */

    // first check if the tangent is a unit vector
    Tensor<1,dim> t = tangent;
    if (abs(t.norm() - 1.0) > DefaultValues::small_number)
      t = t/t.norm();

    const double d = cell->diameter();
    const double step = d/100;

    double length = 0;
    // first move in the direction of a
    Point<dim> p = start, pp = start;
    while (cell->point_inside(p))
    {
      length += (p - pp).norm();
      pp = p;
      p = p + t*step;
    } // end moving along tangent

    // then move in the opposite direction
    p = start, pp = start;
    while (cell->point_inside(p))
      {
        length += (p - pp).norm();
        pp = p;
        p = p - t*step;
      } // end moving along tangent

    return length;
  }  // eom

  template <int dim>
  const std::vector< Point<dim> > &
  Wellbore<dim>::get_locations()
  {
    return locations;
  }
}  // end of namespace

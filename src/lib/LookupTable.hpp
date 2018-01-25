#pragma once

#include <deal.II/lac/vector.h>
#include <algorithm>  // is_sorted
#include <deal.II/lac/full_matrix.h>


namespace Interpolation
{
using namespace dealii;


class LookupTable
{
 public:
  LookupTable(const bool interpolate_=true,
              const bool extrapolate_=false);
  LookupTable(const Vector<double>     &x,
              const FullMatrix<double> &y,
              const bool           interpolate_  =true,
              const bool           extrapolate_=true);
  LookupTable(const FullMatrix<double> &xy,
              const bool               interpolate_=true,
              const bool               extrapolate_=true);
  void set_data(const FullMatrix<double> &xy);
  // getting data
  double get_value(const double x,
                   const int    col) const;
  void get_values(const double           x,
                  const std::vector<int> &cols,
                  std::vector<double>    &dst) const;
  void get_values(const double           x,
                  std::vector<double>    &dst) const;
  void get_values_and_derivatives(const double           x,
                                  const std::vector<int> &cols,
                                  const std::vector<int> &cols_d,
                                  std::vector<double>    &dst) const;
 private:
  Vector<double> x_values;
  FullMatrix<double>     y_values;
  bool interpolate, extrapolate;
  std::vector<int> all_columns;
};

LookupTable::LookupTable(const bool interpolate_,
                         const bool extrapolate_)
    :
    interpolate(interpolate_),
    extrapolate(extrapolate_)
{}


LookupTable::LookupTable(const Vector<double>     &x,
                         const FullMatrix<double> &y,
                         const bool               interpolate_,
                         const bool               extrapolate_)
    :
    x_values(x),
    y_values(y),
    interpolate(interpolate_),
    extrapolate(extrapolate_)
{
  AssertThrow(y.m() == x.size(),
              ExcDimensionMismatch(x.size(), y.m()));
  AssertThrow(x.size() > 0,
              ExcMessage("vector is empty"));
  AssertThrow(y.n() > 0,
              ExcMessage("matrix empty"));
  AssertThrow(std::is_sorted(x.begin(), x.end()),
              ExcMessage("x should be sorted"));
  all_columns.resize(y.n());
}


LookupTable::LookupTable(const FullMatrix<double> &xy,
                         const bool               interpolate_,
                         const bool               extrapolate_)
    :
    x_values(xy.m()),
    y_values(xy.m(),xy.n()-1),
    interpolate(interpolate_),
    extrapolate(extrapolate_)
{
  set_data(xy);
} // eom

void LookupTable::set_data(const FullMatrix<double> &xy)
{
  // AssertThrow(xy.m() > 0,
  //             ExcEmptyMatrix());
  AssertThrow(xy.n() > 1, ExcMessage("Need at least 2 columns"));
  if (x_values.size() != xy.m())
    x_values.reinit(xy.m());
  if (y_values.m() != xy.m() || y_values.n() != xy.n()-1)
    y_values.reinit(xy.m(), xy.n()-1);

  for (unsigned int i=0; i<xy.m(); ++i)
    x_values[i] = xy(i, 0);
  y_values.fill(xy, 0, 0, 0, 1);

  all_columns.resize(y_values.n());
  for (unsigned int i=0; i<all_columns.size(); ++i)
    all_columns[i] = i;
}


double LookupTable::get_value(const double x,
                              const int    col) const
{
  std::vector<double> dst = {0.0};
  std::vector<int> cols = {col};
  get_values(x, cols, dst);
  const double value = dst[0];
  return value;
}  // eom



void LookupTable::get_values(const double           x,
                             const std::vector<int> &cols,
                             std::vector<double>    &dst) const

{
  const std::vector<int> cols_d;
  get_values_and_derivatives(x, cols, cols_d, dst);
}  // eom



void LookupTable::get_values(const double           x,
                             std::vector<double>    &dst) const
{
  get_values(x, all_columns, dst);
} // eom



void LookupTable::
get_values_and_derivatives(const double           x,
                           const std::vector<int> &cols,
                           const std::vector<int> &cols_d,
                           std::vector<double>    &dst) const

{
  for (const auto & col : cols)
    AssertThrow(static_cast<unsigned int>(col) < y_values.n(),
                ExcDimensionMismatch(col, y_values.n()));
  for (const auto & col : cols_d)
    AssertThrow(static_cast<unsigned int>(col) < y_values.n(),
                ExcDimensionMismatch(col, y_values.n()));

  AssertThrow(cols.size() + cols_d.size() == dst.size(),
              ExcDimensionMismatch(cols.size() + cols_d.size(), dst.size()));

  const unsigned int size = x_values.size();
  // const unsigned int n_col = cols.size();

  if (size == 1) // case with constant value
  {
    // y_values.print_formatted(std::cout, 3, false);
    // for (unsigned int shit=0; shit<cols.size(); ++shit)
    //   std::cout << "col " << cols[shit] << std::endl;
    for (unsigned int c=0; c<cols.size(); ++c)
      dst[c] = y_values(0, cols[c]);
    for (const auto & c:cols_d)
      dst[cols.size()+c] = 0;
    return;
  }

  int i = 0;                      // find left end of interval for interpolation
  if ( x >= x_values[size - 2] )     // special case: beyond right end
    i = size - 2;
  else
  {
    while ( x > x_values[i+1] )
      i++;
  }

  double xL = x_values[i];
  double xR = x_values[i+1];

  for (unsigned int c=0; c<cols.size(); ++c)
  {
    // points on either side (unless beyond ends)
    double yL = y_values(i, c);
    double yR = y_values(i+1, c);

    if ( !extrapolate )  // if beyond ends of array and not extrapolating
    {
      if ( x < xL ) yR = yL;
      if ( x > xR ) yL = yR;
    }

    if (interpolate)
    {
      const double dydx = ( yR - yL ) / ( xR - xL );  // gradient
      dst[c] = yL + dydx*( x - xL );                  // linear interpolation
    }
    else
      dst[c] = yL;                                    // lookup
  } // end loop for values

  // get inverse derivatives
  for (unsigned int c=0; c<cols_d.size(); ++c)
  {
    // points on either side (unless beyond ends)
    double yL = y_values(i, c);
    double yR = y_values(i+1, c);

    if ( !extrapolate )  // if beyond ends of array and not extrapolating
    {
      if ( x < xL ) yR = yL;
      if ( x > xR ) yL = yR;
    }
    dst[cols.size()+c]=( yR - yL ) / ( xR - xL );
  } // end loop for derivatives

} // eom

}

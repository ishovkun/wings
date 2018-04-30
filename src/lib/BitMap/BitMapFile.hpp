#pragma once

// #include <iostream>     // std::cout
#include <fstream>

#include <deal.II/base/tensor.h>

#include <Parsers.hpp>

namespace Wings
{


namespace BitMap {
using namespace dealii;


// template <int dim>
class BitMapFile
{
 public:
  BitMapFile(const std::string &name);
  double get_value(const double x) const;
  double get_value(const double x,
                   const double y) const;
  double get_value(const double x,
                   const double y,
                   const double z) const;
  void scale_coordinates(const double scale);

 private:
  std::vector<double> bitmap_data;
  double hx, hy, hz;
  double maxvalue = 255;
  int nx, ny, nz;
  std::vector<double> dimensions;
  unsigned int dim;
  double get_pixel_value(const int i, const int j) const;
  double get_pixel_value(const int i,
                         const int j,
                         const int k) const;
};


// template <int dim>
// BitMapFile<dim>::BitMapFile(const std::string &name)
BitMapFile::BitMapFile(const std::string &name)
    :
    bitmap_data(0),
    hx(0),
    hy(0),
    hz(0),
    nx(0),
    ny(0),
    nz(0)
{
  std::ifstream f(name.c_str());
  AssertThrow (f, ExcMessage (std::string("Can't read from file <") +
                              name + ">!"));

  std::string temp;

  // Read dimensions
  getline(f, temp);
  // std::cout << temp << std::endl;

  dimensions = Parsers::parse_string_list<double>(temp, "\t ");
  // std::cout << "tmp "<< temp << std::endl;
  // std::cout << "dim size "<< dimensions.size() << std::endl;
  // for (unsigned int ddd=0; ddd<dimensions.size(); ++ddd)
  //   std::cout << dimensions[ddd] << std::endl;

  dim = dimensions.size()/2;
  // AssertThrow(dimensions.size()%2 == 0,
  //             ExcMessage("Wrong dimensions in bitmap file"));
  AssertThrow(dimensions.size() == 6,
              ExcMessage("Wrong dimensions in bitmap file"));

  // Read numbers of pixels
  getline(f, temp);
  std::vector<unsigned int> n_pixels =
      Parsers::parse_string_list<unsigned int>(temp, "\t ");

  AssertThrow(n_pixels.size() == dim,
              ExcMessage("Wrong grid size in permeability file"));

  // Calculate total number of pixels
  unsigned int total_pixels = 1;
  for (unsigned int d=0; d<dim; d++)
  {
    total_pixels *= n_pixels[d];
    AssertThrow(n_pixels[d] > 0,  ExcMessage("Invalid file format."));
  }

  // Read values
  bitmap_data.resize(total_pixels);
  for (unsigned int i=0; i<total_pixels; i++)
  {
    AssertThrow(!f.eof(), ExcMessage("Insufficient amount of values"));
    double x;
    f >> x;
    AssertThrow(x > 0,  ExcMessage("Invalid entry."));
    bitmap_data[i] = x;
  }

  nx = n_pixels[0];
  hx = 1.0 / (nx - 1);
  switch (dim)
  {
    case 2:
      ny = n_pixels[1];
      hy = 1.0 / (ny - 1);
      nz = 1;
      hz = 0.0;
      break;
    case 3:
      ny = n_pixels[1];
      nz = n_pixels[2];
      hy = 1.0 / (ny - 1);
      if (nz != 1)
        hz = 1.0 / (nz - 1);
      else
        hz = 0;
      break;
  }
}  // eom



double BitMapFile::get_pixel_value(const int i,
                                   const int j) const
{
  // std::cout << "i = " << i << "\tj = " << j  << std::endl;
  assert(i >= 0 && i < nx);
  assert(j >= 0 && j < ny);
  return bitmap_data[nx * (ny - 1 - j) + i];
}  // eom



double BitMapFile::get_pixel_value(const int i,
                                   const int j,
                                   const int k) const
{
  // std::cout << "i = " << i << "\tj = " << j << "\tk = " << k << std::endl;
  assert(i >= 0 && i < nx);
  assert(j >= 0 && j < ny);
  assert(k >= 0 && k < nz);
  // return bitmap_data[nx * (ny - 1 - j) + i];
  return bitmap_data[nx*ny*k + nx*j + i];
}  // eom



double BitMapFile::get_value(const double x) const
{
  // normalized x and y
  const double xn = (x-dimensions[0])/(dimensions[3]-dimensions[0]);
  // pixel numbers
  const int ix = std::min(std::max((int) (xn / hx), 0), nx - 2);
  // normalized coordinates in unit square
  const double xi  = std::max(std::min((xn-ix*hx)/hx, 1.), 0.);
  // std::cout << "xn " << xn << std::endl;
  // std::cout << "ix " << ix << std::endl;
  // std::cout << "xi " << xi << std::endl;
  // std::cout << "hx " << hx << std::endl;
  // std::cout << "exp " << (xn-ix*hx)/hx << std::endl;
  // linear interpolation
  return ((1-xi)*get_pixel_value(ix, 0)
          +
          xi*get_pixel_value(ix+1, 0));
}  // eom



double BitMapFile::get_value(const double x,
                             const double y) const
{
  // normalized x and y
  const double xn = (x-dimensions[0])/(dimensions[3]-dimensions[0]);
  const double yn = (y-dimensions[1])/(dimensions[4]-dimensions[1]);
  // pixel numbers
  const int ix = std::min(std::max((int) (xn / hx), 0), nx - 2);
  const int iy = std::min(std::max((int) (yn / hy), 0), ny - 2);
  // normalized coordinates in unit square
  // const double xi  = std::min(std::max((xn-ix*hx)/hx, 1.), 0.);
  // const double eta = std::min(std::max((yn-iy*hy)/hy, 1.), 0.);
  const double xi  = std::max(std::min((xn-ix*hx)/hx, 1.), 0.);
  const double eta = std::max(std::min((yn-iy*hy)/hy, 1.), 0.);
  // bilinear interpolation
  return ((1-xi)*(1-eta)*get_pixel_value(ix,iy)
          +
          xi*(1-eta)*get_pixel_value(ix+1,iy)
          +
          (1-xi)*eta*get_pixel_value(ix,iy+1)
          +
          xi*eta*get_pixel_value(ix+1,iy+1));
}  // eom



double BitMapFile::get_value(const double x,
                             const double y,
                             const double z) const
{
  if (ny == 1 && nz == 1)
    return get_value(x);
  if (nz == 1)
    return get_value(x, y);

  // normalized x and y
  const double xn = (x-dimensions[0])/(dimensions[3]-dimensions[0]);
  const double yn = (y-dimensions[1])/(dimensions[4]-dimensions[1]);
  const double zn = (z-dimensions[2])/(dimensions[5]-dimensions[2]);
  // pixel numbers
  const int ix = std::min(std::max((int) (xn / hx), 0), nx - 2);
  const int iy = std::min(std::max((int) (yn / hy), 0), ny - 2);
  const int iz = std::min(std::max((int) (zn / hz), 0), ny - 2);
  // normalized coordinates in unit cube
  // const double xi  = std::min(std::max((xn-ix*hx)/hx, 1.), 0.);
  // const double eta = std::min(std::max((yn-iy*hy)/hy, 1.), 0.);
  // const double zeta = std::min(std::max((zn-iz*hz)/hz, 1.), 0.);
  const double xi  = std::max(std::min((xn-ix*hx)/hx, 1.), 0.);
  const double eta = std::max(std::min((yn-iy*hy)/hy, 1.), 0.);
  const double zeta = std::max(std::min((zn-iz*hz)/hz, 1.), 0.);
  // trilinear interpolation
  return
      (1-xi)*(1-eta)*(1-zeta)*get_pixel_value(ix, iy, iz)
      +
      xi*(1-eta)*(1-zeta)*get_pixel_value(ix+1, iy, iz)
      +
      (1-xi)*eta*(1-zeta)*get_pixel_value(ix, iy+1, iz)
      +
      (1-xi)*(1-eta)*zeta*get_pixel_value(ix, iy, iz+1)
      +
      xi*(1-eta)*zeta*get_pixel_value(ix+1, iy, iz+1)
      +
      (1-xi)*eta*zeta*get_pixel_value(ix, iy+1, iz+1)
      +
      xi*eta*(1-zeta)*get_pixel_value(ix+1, iy+1, iz)
      +
      xi*eta*zeta*get_pixel_value(ix+1, iy+1, iz+1);
}  // eom



void BitMapFile::scale_coordinates(const double scale)
{
  for (auto & dimension : dimensions)
    dimension = dimension * scale;
}  // eom



}  // end of namespace


}  // end Wings

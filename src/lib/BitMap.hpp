#pragma once

// #include <iostream>     // std::cout
#include <fstream>

#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>

#include <Parsers.hpp>


namespace BitMap {
  using namespace dealii;

  template <int dim>
  class BitMapFile
  {
  public:
    BitMapFile(const std::string &name);
    double get_value(const double x, const double y) const;

  private:
    std::vector<double> bitmap_data;
    double hx, hy, hz;
    double maxvalue = 255;
    int nx, ny, nz;
    std::vector<double> dimensions;
    double get_pixel_value(const int i, const int j) const;
  };


  template <int dim>
  BitMapFile<dim>::BitMapFile(const std::string &name)
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
    // std::cout << "n dim " << dimensions.size() << std::endl;
    AssertThrow(dimensions.size() == 2*dim,
                ExcMessage("Wrong dimensions in permeability file"));
    // for (int i=0; i<dimensions.size(); ++i)
    //   AssertThrow(dimensions[i] > 0,  ExcMessage("Invalid file format."));

    // Read numbers of pixels
    getline(f, temp);
    std::vector<unsigned int> n_pixels =
      Parsers::parse_string_list<unsigned int>(temp, "\t ");
    AssertThrow(n_pixels.size() == dim,
                ExcMessage("Wrong grid size in permeability file"));
    // Calculate total number of pixels
    unsigned int total_pixels = 1;
    for (int d=0; d<dim; d++)
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
      // std::cout << x << std::endl;
      // std::cout << !f.eof() << std::endl;
    }

    nx = n_pixels[0];
    hx = 1.0 / (nx - 1);
    switch (dim)
    {
      case 2:
        ny = n_pixels[1];
        hy = 1.0 / (ny - 1);
        break;
      case 3:
        ny = n_pixels[1];
        nz = n_pixels[2];
        hy = 1.0 / (ny - 1);
        hz = 1.0 / (nz - 1);
        break;
    }
  }  // eom


  template <>
  double BitMapFile<2>::get_pixel_value(const int i,
                                        const int j) const
  {
    assert(i >= 0 && i < nx);
    assert(j >= 0 && j < ny);
    return bitmap_data[nx * (ny - 1 - j) + i];
  }  // eom

  template <>
  double BitMapFile<2>::get_value(const double x,
                                  const double y) const
  {
    // normalized x and y
    const double xn = (x-dimensions[0])/(dimensions[2]-dimensions[0]);
    const double yn = (y-dimensions[1])/(dimensions[3]-dimensions[1]);
    // pixel numbers
    const int ix = std::min(std::max((int) (xn / hx), 0), nx - 2);
    const int iy = std::min(std::max((int) (yn / hy), 0), ny - 2);
    // some relative location... I'm not sure
    const double xi  = std::min(std::max((xn-ix*hx)/hx, 1.), 0.);
    const double eta = std::min(std::max((yn-iy*hy)/hy, 1.), 0.);

    return ((1-xi)*(1-eta)*get_pixel_value(ix,iy)
            +
            xi*(1-eta)*get_pixel_value(ix+1,iy)
            +
            (1-xi)*eta*get_pixel_value(ix,iy+1)
            +
            xi*eta*get_pixel_value(ix+1,iy+1));
  }  // eom


  template <int dim>
  class BitMapFunction : public Function<dim>
  {
  public:
    BitMapFunction(const std::string &filename);
    BitMapFunction(const std::string   &filename,
                   const Tensor<1,dim> &anisotropy_);

    double value(const Point<dim> &p,
                 const unsigned int /*component*/ c) const;
    void vector_value(const Point<dim> &p,
                      Tensor<1,dim>    &v) const;
    void vector_value(const Point<dim> &p,
                      Vector<double>   &v) const;
  private:
    BitMapFile<dim> f;
    Tensor<1,dim> anisotropy;
  };


  template<int dim>
  BitMapFunction<dim>::BitMapFunction(const std::string &filename)
    :
    Function<dim>(1),
    f(filename)
  {
    for (int d=0; d<dim; d++)
      anisotropy[d] = 1;
  }  // eom


  template<int dim>
  BitMapFunction<dim>::BitMapFunction(const std::string   &filename,
                                      const Tensor<1,dim> &anisotropy_)
    :
    Function<dim>(1),
    f(filename),
    anisotropy(anisotropy_)
  {}  // eom


  template<int dim>
  void
  BitMapFunction<dim>::vector_value(const Point<dim> &p,
                                    Tensor<1,dim>    &v) const
  {
    AssertThrow(v.size() < dim, ExcMessage("Wrong dimensions"));
    for (int c=0; c<dim; c++)
      v[c] = value(p, c);
  }  // eom


  template<int dim>
  void
  BitMapFunction<dim>::vector_value(const Point<dim> &p,
                                    Vector<double>   &v) const
  {
    for (int c=0; c<dim; c++)
      v[c] = value(p, c);
  }  // eom


  template<>
  double
  BitMapFunction<2>::value(const Point<2> &p,
                           const unsigned int /*component*/ c) const
  {
    Assert(c<2, ExcNotImplemented());
    return f.get_value(p(0),p(1))*anisotropy[c];
  }  // eom

}  // end of namespace

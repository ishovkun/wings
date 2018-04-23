#pragma once

#include <Model.hpp>

namespace Wings {

static const int dim = 3;

class Probe
{
 public:
  Probe(const Model<dim> & model);
  ~Probe();

 private:
  const Model<dim> & model;
};

Probe(const Model & model)
    :
    model(model)
{}

} // end wings

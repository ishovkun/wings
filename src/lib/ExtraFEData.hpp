#pragma once

namespace ExtraFEData
{
using namespace dealii;

enum FEDerivativeOrder {values, divergence};

template <int dim>
class ExtraFEData
{
 public:
  ExtraFEData(const std::vector<unsigned int> vectors_per_dof_handler,
              const unsigned int              n_quadrature_points=1);
  ~ExtraFEData();
  void
  set_data(std::vector< const DoFHandler<dim>* >                      &dof_handlers_,
           std::vector< std::vector<TrilinosWrappers::MPI::Vector*> > &vectors_,
           std::vector< std::vector<FEDerivativeOrder> > &derivative_orders_);
  void make_fe_values(QGauss<dim> &quadrature_formula);
  void cells_begin();
  void increment_cells();
  void reinit();
  void update_fe_values();
  const std::vector<double> & get_values(const unsigned int q_point);

  // attributes
  std::vector< const DoFHandler<dim>* > dof_handlers;
  std::vector< std::vector<TrilinosWrappers::MPI::Vector*> > vectors;
  std::vector< std::vector<FEDerivativeOrder> > derivative_orders;
  std::vector< std::vector< std::vector<double> > > values;
  // this is the same as values but stored in an easily accessible format
  std::vector<double> unpacked_values;
  std::vector< FEValues<dim>* > fe_values;
  std::vector< typename DoFHandler<dim>::active_cell_iterator > cells;
 private:
  void clear_fe_values();
  unsigned int n_quadrature_points;
};



template <int dim>
ExtraFEData<dim>::
ExtraFEData(const std::vector<unsigned int> vectors_per_dof_handler,
            const unsigned int              n_quadrature_points)
    :
    dof_handlers(vectors_per_dof_handler.size()),
    vectors(vectors_per_dof_handler.size()),
    derivative_orders(vectors_per_dof_handler.size()),
    values(vectors_per_dof_handler.size())
{
  const unsigned int n_dof_handlers = vectors_per_dof_handler.size();
  this->n_quadrature_points = n_quadrature_points;
  for (unsigned int i=0; i<n_dof_handlers; ++i)
  {
    const unsigned int n_vectors = vectors_per_dof_handler[i];
    unsigned int n_values = 0;
    vectors[i].resize(n_vectors);
    derivative_orders[i].resize(n_vectors);
    values[i].resize(n_vectors);
    for (unsigned int j=0; j<n_vectors; ++j)
    {
      values[i][j].resize(n_quadrature_points);
      n_values += n_quadrature_points;
    }
    unpacked_values.resize(n_values);
  }
}  // eom



template <int dim>
void ExtraFEData<dim>::
set_data(std::vector< const DoFHandler<dim>* >                      &dof_handlers_,
         std::vector< std::vector<TrilinosWrappers::MPI::Vector*> > &vectors_,
         std::vector< std::vector<FEDerivativeOrder> > &derivative_orders_)
{
  for (unsigned int i=0; i < dof_handlers.size(); ++i)
  {
    dof_handlers[i] = dof_handlers_[i];
    for (unsigned int j=0; j<vectors[i].size(); ++j)
    {
      vectors[i][j] = vectors_[i][j];
      derivative_orders[i][j] = derivative_orders_[i][j];
    }
  }
}  // eom



template <int dim>
void ExtraFEData<dim>::
make_fe_values(QGauss<dim> &quadrature_formula)
{
  clear_fe_values();
  fe_values.resize(dof_handlers.size());

  for (unsigned int i=0; i<dof_handlers.size(); ++i)
  {
    const auto & fe = dof_handlers[i]->get_fe();
    fe_values[i] =
        new FEValues<dim>(fe, quadrature_formula, update_values);
  }
}  // eom



template <int dim>
void ExtraFEData<dim>::clear_fe_values()
{
  for (auto & fe_val: fe_values)
    delete fe_val;
  fe_values.clear();
}  // eom



template <int dim>
ExtraFEData<dim>::~ExtraFEData()
{
  clear_fe_values();
}


template <int dim>
void ExtraFEData<dim>::cells_begin()
{
  cells.resize(dof_handlers.size());
  for (unsigned int i=0; i<dof_handlers.size(); ++i)
    cells[i] = dof_handlers[i]->begin_active();
} // eom



template <int dim>
void ExtraFEData<dim>::increment_cells()
{
  for (auto & cell : cells)
    cell++;
} // eom



template <int dim>
void ExtraFEData<dim>::reinit()
{
  for (unsigned int i=0; i< fe_values.size(); ++i)
    fe_values[i]->reinit(cells[i]);
} // eom



template <int dim>
void ExtraFEData<dim>::update_fe_values()
{
  for (unsigned int i=0; i<fe_values.size(); ++i)
  {
    for (unsigned int j=0; j<values[i].size(); ++j)
    {
      if (derivative_orders[i][j] == FEDerivativeOrder::values)
        fe_values[i]->get_function_values(*(vectors[i][j]), values[i][j]);
      else
        AssertThrow(false, ExcNotImplemented());
    }
  }
} // eom



template<int dim>
const std::vector<double> &
ExtraFEData<dim>::get_values(const unsigned int q_point)
{
  for (unsigned int i=0; i<values.size(); ++i)
    for (unsigned int j=0; j<values.size(); ++j)
      unpacked_values[i+j] = values[i][j][q_point];
  return unpacked_values;
} // eom

} // end of namespace

#pragma once
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <deal.II/base/conditional_ostream.h>

// Custom modules
#include <Wellbore.hpp>
#include <Parsers.hpp>
#include <BitMap/BitMapFunction.hpp>
#include <BitMap/BitMapTensorFunction.hpp>
#include <Units.h>
#include <Tensors.hpp>
#include <Keywords.h>
#include <SyntaxParser.hpp>
#include <Model.hpp>
#include <LookupTable.hpp>
#include <Math.hpp>

namespace Parsers {

constexpr int dim = 3;

  class Reader
  {
  public:
    Reader(ConditionalOStream & pcout_,
           Model::Model<dim>  & model_);
    void read_input(const std::string&,
                    const int verbosity_=0);
    void print_input();

  protected:
    void read_file(const std::string& fname);
    void assign_wells(const std::string  &kwd,
                      const SyntaxParser &parser);
    void assign_schedule(const std::string  &kwd,
                         const SyntaxParser &parser);
    Function<dim> *
    get_function(const std::string   & kwd,
                 const SyntaxParser  & parser) const;

    TensorFunction<2,dim,double> *
    get_tensor_function(const std::string   & kwd,
                        const Tensor<1,dim> & anisotropy,
                        const SyntaxParser  & parser) const;

    ConditionalOStream & pcout;
    Model::Model<dim>  & model;
    int                 verbosity;
    std::string         input_text;
    std::string         input_file_name;
  }; //


  Reader::Reader(ConditionalOStream & pcout_,
                 Model::Model<dim>  & model_)
    :
    pcout(pcout_),
    model(model_),
    verbosity(0),
    input_file_name("")
  {}  // eom


  void
  Reader::read_file(const std::string& fname)
  {
    std::ifstream t(fname);
    std::stringstream buffer;
    buffer << t.rdbuf();
    input_text = buffer.str();
  } // eom


void
Reader::read_input(const std::string& fname,
                   const int verbosity_)
{
  verbosity = verbosity_;
  input_file_name = fname;
  read_file(fname);
  Parsers::strip_comments(input_text, "#");
  if (verbosity > 1)
    std::cout << input_text << std::endl;
  // Keywords::Keywords kwds;
  SyntaxParser parser(input_text);
  { // Mesh
    parser.enter_subsection(Keywords::section_mesh);
    model.initial_refinement_level =
        parser.get_int(Keywords::global_refinement_steps, 0);
    model.n_adaptive_steps =
        parser.get_int(Keywords::adaptive_refinement_steps, 0);

    auto tri_str =
        parser.get_str_list(Keywords::triangulation, std::string("\t "));

    if (tri_str.size() == 9) // create mesh
    {
      model.mesh_config.type = Model::MeshType::Create;
      for (int i=0; i<dim; ++i)
        model.mesh_config.n_cells[i] = Parsers::convert<int>(tri_str[i]);

      Point<dim> p1, p2;
      p1[0] = Parsers::convert<double>(tri_str[3]);
      p1[1] = Parsers::convert<double>(tri_str[4]);
      p1[2] = Parsers::convert<double>(tri_str[5]);
      p2[0] = Parsers::convert<double>(tri_str[6]);
      p2[1] = Parsers::convert<double>(tri_str[7]);
      p2[2] = Parsers::convert<double>(tri_str[8]);
      model.mesh_config.points = std::make_pair(p1, p2);
    }
    else if (tri_str.size() == 2)
    {
      if (tri_str[0] == Keywords::file_msh)
        model.mesh_config.type = Model::MeshType::Msh;
      else if (tri_str[0] == Keywords::file_abaqus)
        model.mesh_config.type = Model::MeshType::Abaqus;
      else
        AssertThrow(false, ExcMessage("Wrong Triangulation entry"));

      // std::cout <<
      model.mesh_config.file =
          boost::filesystem::path(fname).parent_path() / tri_str[1];
    }
    else
    {
      AssertThrow(false, ExcMessage("Wrong Triangulation entry"));
    }
    // model.mesh_file =
    //   boost::filesystem::path(fname).parent_path() /
    //   parser.get(Keywords::mesh_file);
    // std::cout << model.mesh_file << std::endl;
  } // end section mesh
  { // initial conditions
    parser.enter_subsection(Keywords::section_initial_conditions);
    // reference pressure
    const double ref_p =
        parser.get_double(Keywords::reference_pressure);  // required
    model.reference_pressure = ref_p*model.units.pressure();
    // TODO: reference depth
    // Initial saturation
    if(model.n_phases() > 1)
    {
      const double sw_init =
          parser.get_double(Keywords::initial_saturation_water);
      model.initial_saturation_water = sw_init;
    }
  } // end section init conditions
  {  // equation data
    parser.enter_subsection(Keywords::section_equation_data);
    std::string model_type_str = parser.get(Keywords::model_type);
    // std::cout << model_type_str << std::endl;

    Model::FluidModelType model_type(Model::FluidModelType::Liquid);
    if (model_type_str == Keywords::model_single_liquid)
      model_type = Model::FluidModelType::Liquid;
    else if (model_type_str == Keywords::model_water_oil)
      model_type = Model::FluidModelType::DeadOil;
    else
      AssertThrow(false, ExcMessage("Wrong entry in " + Keywords::model_type));

    model.set_fluid_model(model_type);

    { // units
      const auto & tmp = parser.get(Keywords::unit_system);
      if (tmp == Keywords::si_units)
        model.units.set_system(Units::UnitSystem::si_units);
      else if (tmp == Keywords::field_units)
        model.units.set_system(Units::UnitSystem::field_units);
    }

    {// Permeability & porosity
      std::vector<double> default_anisotropy{1,1,1};
      Tensor<1,dim> anisotropy = Parsers::convert<dim>
          (parser.get_number_list<double>(Keywords::permeability_anisotropy, ",",
                                          default_anisotropy));
      // apply units
      anisotropy *= model.units.permeability();

      model.get_permeability =
          get_tensor_function(Keywords::permeability, anisotropy, parser);

      model.get_porosity =
          get_function(Keywords::porosity, parser);
    }

    if (model.has_phase(Model::Phase::Water))
    {
      auto tmp = parser.get_matrix(Keywords::pvt_water, ";", ",");
      // tmp.print_formatted(std::cout);
      AssertThrow(tmp.n() == model.n_pvt_water_columns,
                  ExcDimensionMismatch(tmp.n(), model.n_pvt_water_columns));
      // loop through columns and apply units
      for (unsigned int i=0; i<tmp.m(); ++i)
      {
        tmp(i, 0) *= model.units.pressure();
        tmp(i, 2) *= model.units.compressibility();
        tmp(i, 3) *= model.units.viscosity();
      }
      Interpolation::LookupTable pvt_water_table(tmp);
      model.set_pvt_water(pvt_water_table);
      // density
      double rho_w = parser.get_double(Keywords::density_sc_water);
      rho_w *= model.units.density();
      model.set_density_sc_w(rho_w);
    }

    if (model.has_phase(Model::Phase::Oleic))
    {
      auto tmp = parser.get_matrix(Keywords::pvt_oil, ";", ",");
      // deadoil: p Bo Co mu_o
      AssertThrow(tmp.n() == model.n_pvt_oil_columns,
                  ExcDimensionMismatch(tmp.n(), model.n_pvt_oil_columns));
      // loop through columns and apply units
      for (unsigned int i=0; i<tmp.m(); ++i)
      {
        tmp(i, 0) *= model.units.pressure();
        tmp(i, 2) *= model.units.compressibility();
        tmp(i, 3) *= model.units.viscosity();
      }
      Interpolation::LookupTable pvt_oil_table(tmp);
      model.set_pvt_oil(pvt_oil_table);

      double rho_o = parser.get_double(Keywords::density_sc_oil);
      rho_o *= model.units.density();
      model.set_density_sc_o(rho_o);
    }  // end two-phase case

    if (model.fluid_model == Model::DeadOil)
    {
      // Relative permeability
      const auto & rel_perm_water =
          parser.get_number_list<double>(Keywords::rel_perm_water, ",");
      const auto & rel_perm_oil =
          parser.get_number_list<double>(Keywords::rel_perm_oil, ",");
      AssertThrow(rel_perm_water.size() == 3,
                  ExcDimensionMismatch(rel_perm_water.size(), 3));
      AssertThrow(rel_perm_oil.size() == 3,
                  ExcDimensionMismatch(rel_perm_oil.size(), 3));
      model.set_rel_perm(rel_perm_water[0], rel_perm_oil[0],
                         rel_perm_water[1], rel_perm_oil[1],
                         rel_perm_water[2], rel_perm_oil[2]);
    }

    { // solid model
      // can be Compressibility, Elasticity, default compressibility
      std::string model_type_str = parser.get(Keywords::solid_model_type,
                                              Keywords::model_compressibility);

      // Model::SolidModelType solid_model_type(Model::SolidModelType::Compressibility);
      if (model_type_str == Keywords::model_compressibility)
        model.set_solid_model(Model::SolidModelType::Compressibility);
      else if (model_type_str == Keywords::model_elasticity)
        model.set_solid_model(Model::SolidModelType::Elasticity);
      else
        AssertThrow(false, ExcNotImplemented());

      if (model.solid_model == Model::SolidModelType::Compressibility)
      {
        const double c_rock = parser.get_double(Keywords::rock_compressibility, 0.0);
        model.set_rock_compressibility(c_rock);
      }
      else if (model.solid_model == Model::SolidModelType::Elasticity)
      {
        // biot coefficient, young's modulus (func), Poisson_ratio (function)
        const double biot_coef = parser.get_double(Keywords::biot_coefficient);
        model.set_biot_coefficient(biot_coef);
        model.get_young_modulus =
            get_function(Keywords::young_modulus, parser);
        model.get_poisson_ratio =
            get_function(Keywords::poisson_ratio, parser);
      }
      else
      {
        AssertThrow(false, ExcNotImplemented());
      }
    }

  } // end equation data
  { // boundary conditions: aquifers, solid bc's
    try {
      parser.enter_subsection(Keywords::section_boundary_conditions);
    }
    catch (...)
    {
      AssertThrow(model.solid_model == Model::SolidModelType::Compressibility,
                  ExcMessage("Need solid boundary conditions"));
    }
    /*
     * By default all fluid BC's are no-flux
     */
    if (model.solid_model == Model::SolidModelType::Elasticity)
    {
      // Dirichlet BCs
      const auto dirichlet_labels =
          parser.get_number_list<int>(Keywords::solid_dirichlet_labels,
                                      std::string("\t "));
      const auto dirichlet_components =
          parser.get_number_list<int>(Keywords::solid_dirichlet_components,
                                      std::string("\t "));
      auto dirichlet_values =
          parser.get_number_list<double>(Keywords::solid_dirichlet_values,
                                         std::string("\t "));
      for (unsigned int i=0; i < dirichlet_values.size(); ++i)
        dirichlet_values[i] *= model.units.length();

      AssertThrow(dirichlet_labels.size() > 0,
                  ExcMessage("Need at least one Dirichlet boundary"));
      AssertThrow(dirichlet_labels.size() == dirichlet_values.size()
                  &&
                  dirichlet_values.size() == dirichlet_components.size(),
                  ExcMessage("Inconsistent displacement boundary conditions"));
      for (const auto & comp : dirichlet_components)
        AssertThrow(comp < 3 && comp >= 0, ExcDimensionMismatch(comp, dim));

      model.set_solid_dirichlet_boundary_conditions(dirichlet_labels,
                                                    dirichlet_components,
                                                    dirichlet_values);
      // Neumann BC's -- not required
      const auto neumann_labels =
          parser.get_number_list<int>(Keywords::solid_neumann_labels,
                                      std::string("\t "), std::vector<int>());
      const auto neumann_components =
          parser.get_number_list<int>(Keywords::solid_neumann_components,
                                      std::string("\t "), std::vector<int>());
      auto neumann_values =
          parser.get_number_list<double>(Keywords::solid_neumann_values,
                                         std::string("\t "), std::vector<double>());
      // scale by units of pressure
      for (unsigned int i=0; i<neumann_values.size(); ++i)
        neumann_values[i] *= model.units.pressure();

      AssertThrow(neumann_labels.size() == neumann_values.size() &&
                  neumann_values.size() == neumann_components.size(),
                  ExcMessage("Inconsistent stress boundary conditions"));

      for (const auto & comp : dirichlet_components)
        AssertThrow(comp < 3 && comp >= 0, ExcDimensionMismatch(comp, dim));

      model.set_solid_neumann_boundary_conditions(neumann_labels,
                                                  neumann_components,
                                                  neumann_values);
    }
  }

  {  // wells
    parser.enter_subsection(Keywords::section_wells, /* required = */ false);
    assign_wells(Keywords::well_parameters, parser);
    assign_schedule(Keywords::well_schedule, parser);
  }
  {  // solver
    parser.enter_subsection(Keywords::section_solver);
    model.min_time_step =
        parser.get_double(Keywords::minimum_time_step, 1e-10) *
        model.units.time();
    model.t_max =
        parser.get_double(Keywords::t_max) *
        model.units.time();

    // linear solvers
    // fluid
    const std::string fluid_linear_solver =
        parser.get(Keywords::fluid_linear_solver, Keywords::linear_solver_cg);
    if (fluid_linear_solver == Keywords::linear_solver_direct)
      model.set_fluid_linear_solver(Model::LinearSolverType::Direct);
    else if (fluid_linear_solver == Keywords::linear_solver_cg)
      model.set_fluid_linear_solver(Model::LinearSolverType::CG);
    else if (fluid_linear_solver == Keywords::linear_solver_gmres)
      model.set_fluid_linear_solver(Model::LinearSolverType::GMRES);
    else
      AssertThrow(false, ExcNotImplemented());
    // solid
    const std::string solid_linear_solver =
        parser.get(Keywords::solid_linear_solver, Keywords::linear_solver_cg);
    if (solid_linear_solver == Keywords::linear_solver_direct)
      model.set_solid_linear_solver(Model::LinearSolverType::Direct);
    else if (solid_linear_solver == Keywords::linear_solver_cg)
      model.set_solid_linear_solver(Model::LinearSolverType::CG);
    else
      AssertThrow(false, ExcNotImplemented());

    // tolerance maximum iterations (not required)
    const double coupling_tol = parser.get_double(Keywords::coupling_tolerance, 1e-10);
    const int n_coupling_steps = parser.get_int(Keywords::max_coupling_steps, 20);
    model.coupling_tolerance = coupling_tol;
    model.max_coupling_steps = n_coupling_steps;
  }  // end solver section
} // eom



TensorFunction<2, dim, double> *
Reader::get_tensor_function(const std::string    & kwd,
                            const Tensor<1,dim>  & anisotropy,
                            const SyntaxParser   & parser) const
{
  const auto kwd_list =
      parser.get_str_list(kwd, std::string("\t "));

  Tensor<2,dim> anisotropy_tensor = Math::get_identity_tensor<dim>();
  for (int d=0; d<dim; ++d)
    anisotropy_tensor[d][d] = anisotropy[d];

  std::string entry = kwd_list[0];

  if (entry == "bitmap" && kwd_list.size() == 2)
  { // create bitmap function
    if (verbosity > 0)
      std::cout << "Searching " << kwd_list[1] << std::endl;

    boost::filesystem::path data_file =
        boost::filesystem::path(input_file_name).parent_path() / kwd_list[1];

    BitMap::BitMapTensorFunction<2,dim>* bmf =
        new BitMap::BitMapTensorFunction<2,dim>(data_file.string(),
                                                anisotropy_tensor);

    bmf->scale_coordinates(model.units.length());

    return bmf;
  }
  else if (Parsers::is_number(entry) && kwd_list.size() == 1)
  { // create constant function
    std::vector<double> quantity;
    for (int c=0; c<dim; c++)
      quantity.push_back(boost::lexical_cast<double>(entry)*anisotropy[c]);
    return new ConstantTensorFunction<2,dim,double>(anisotropy_tensor);
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return new ConstantTensorFunction<2,dim,double>(anisotropy_tensor);
}  // eom



Function<dim> *
Reader::get_function(const std::string  & kwd,
                     const SyntaxParser & parser) const
{
  const auto kwd_list =
      parser.get_str_list(kwd, std::string("\t "));

  std::string entry = kwd_list[0];

  if (entry == "bitmap" && kwd_list.size() == 2)
  { // create bitmap function
    if (verbosity > 0)
      std::cout << "Searching " << kwd_list[1] << std::endl;

    boost::filesystem::path data_file =
        boost::filesystem::path(input_file_name).parent_path() / kwd_list[1];

    BitMap::BitMapFunction<dim>* bmf =
        new BitMap::BitMapFunction<dim>(data_file.string());

    bmf->scale_coordinates(model.units.length());

    return bmf;
  }
  else if (Parsers::is_number(entry) && kwd_list.size() == 1)
  { // create constant function
    std::vector<double> quantity;
    for (int c=0; c<dim; c++)
      quantity.push_back(boost::lexical_cast<double>(entry));
    return new ConstantFunction<dim>(quantity);
  }
  else
    AssertThrow(false, ExcNotImplemented());

  return new ConstantFunction<dim>(0);
} // eom


  void Reader::assign_wells(const std::string  &kwd,
                            const SyntaxParser &parser)
  {
    // const auto well_list = parser.get_str_list(kwd, std::string(";"));
    const auto well_list = parser.get_str_list(kwd, std::string(";"),
                                               std::vector<std::string>());
    for (const auto & w : well_list)
    { // loop over individual wells
      // std::cout << w << std::endl;
      std::vector<std::string> well_strs;
      boost::split(well_strs, w, boost::is_any_of(","));
      for (auto & entry : well_strs)
        boost::trim(entry);

      AssertThrow(well_strs.size()>=5,
                  ExcMessage("Wrong entry in well "+well_strs[0]));
      AssertThrow((well_strs.size()-2)%3 == 0,
                  ExcMessage("Wrong entry in well "+well_strs[0]));

      // name
      const std::string name = well_strs[0];
      // radius
      double r = Parsers::convert<double>(well_strs[1]);
      r *= model.units.length();
      // parse locations
      unsigned int n_loc = (well_strs.size()-2) / 3;
      std::vector<Point<dim>> locations(n_loc);
      int loc=0;
      for (unsigned int i=2; i<well_strs.size(); i+=dim)
      {
        double x = Parsers::convert<double>(well_strs[i]);
        double y = Parsers::convert<double>(well_strs[i+1]);
        double z = Parsers::convert<double>(well_strs[i+2]);
        x *= model.units.length();
        y *= model.units.length();
        z *= model.units.length();
        locations[loc] = Point<dim>(x,y,z);
        loc++;
      }

      model.add_well(name, r, locations);

    } // end well loop
  }  // eom

  void Reader::assign_schedule(const std::string  &kwd,
                               const SyntaxParser &parser)
  {
    // const auto lines = parser.get_str_list(kwd, std::string(";"));
    const auto lines = parser.get_str_list(kwd, std::string(";"),
                                           std::vector<std::string>());
    for (auto & line : lines)
    {
      // std::cout << line << std::endl;
      std::vector<std::string> entries;
      boost::algorithm::split(entries, line, boost::is_any_of(","));
      // Handle case when the last entry in schedule ends with ";"
      // Boost thinks that there is something after
      if (entries.size() == 1 && entries[0].size() == 0)
        break;
      // Process entries
      AssertThrow(entries.size() >= 4,
                  ExcMessage("Wrong entry in schedule "+line));

      Schedule::ScheduleEntry schedule_entry;
      // get time
      schedule_entry.time = Parsers::convert<double>(entries[0]);
      // get well name and identifier
      std::string well_name = entries[1];
      // std::cout << "well name = " << well_name << std::endl;
      boost::algorithm::trim(well_name);
      schedule_entry.well_id = model.get_well_id(well_name);
      // get control type
      const int control_type_id = Parsers::convert<int>(entries[2]);
      schedule_entry.control.type =
        Schedule::well_control_type_indexing.find(control_type_id)->second;
      // get control value
      schedule_entry.control.value = Parsers::convert<double>(entries[3]);

      // convert value units
      if (schedule_entry.control.type == Schedule::pressure_control)
      {
        schedule_entry.control.value *= model.units.pressure();
      }

      if (   model.fluid_model == Model::DeadOil
             || model.fluid_model == Model::Liquid)
      {
        if (schedule_entry.control.type != Schedule::pressure_control)
          schedule_entry.control.value *= model.units.fluid_rate();  //
      }
      else if (model.fluid_model == Model::BlackOil)
      {
        if (schedule_entry.control.type == Schedule::flow_control_total
            || schedule_entry.control.type == Schedule::flow_control_phase_1
            || schedule_entry.control.type == Schedule::flow_control_phase_2)
          schedule_entry.control.value *= model.units.fluid_rate();
        if (schedule_entry.control.type == Schedule::flow_control_phase_3)
          schedule_entry.control.value *= model.units.gas_rate();
      }
      else
      {
        AssertThrow(false, ExcNotImplemented());
      }

      model.schedule.add_entry(schedule_entry);
    } // end lines loop
  } // eom
} // end of namespace

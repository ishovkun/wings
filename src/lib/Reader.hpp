#pragma once
// #include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <deal.II/base/conditional_ostream.h>

// Custom modules
#include <Wellbore.hpp>
#include <Parsers.hpp>
#include <BitMap.hpp>
#include <Units.h>
#include <Tensors.hpp>
#include <Keywords.h>
#include <SyntaxParser.hpp>
#include <Model.hpp>
#include <LookupTable.hpp>


namespace Parsers {

  class Reader
  {
  public:
    Reader(ConditionalOStream &pcout_,
           Model::Model<3>    &model_);
    void read_input(const std::string&,
                    const int verbosity_=0);
    void print_input();

  protected:
    void read_file(const std::string& fname);
    void assign_wells(const std::string  &kwd,
                      const SyntaxParser &parser);
    void assign_schedule(const std::string  &kwd,
                         const SyntaxParser &parser);
    Function<3> *
    get_function(const std::string  &kwd,
                 const Tensor<1,3>    &anisotropy,
                 const SyntaxParser &parser);

    ConditionalOStream                     &pcout;
    Model::Model<3>                        &model;
    int                                    verbosity;
    std::string                            input_text;
    std::string                            input_file_name;
  }; //


  Reader::Reader(ConditionalOStream &pcout_,
                 Model::Model<3>    &model_)
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
    Keywords::Keywords kwds;
    SyntaxParser parser(input_text);
    { // Mesh
      parser.enter_subsection(kwds.section_mesh);
      model.initial_refinement_level =
        parser.get_int(kwds.global_refinement_steps, 0);
      model.n_adaptive_steps =
        parser.get_int(kwds.adaptive_refinement_steps, 0);
      model.mesh_file =
        boost::filesystem::path(fname).parent_path() /
        parser.get(kwds.mesh_file);
      // std::cout << model.mesh_file << std::endl;
    }
    {  // equation data
      parser.enter_subsection(kwds.section_equation_data);
      std::string model_type_str = parser.get(kwds.model_type);
      // std::cout << model_type_str << std::endl;

      Model::ModelType model_type(Model::ModelType::SingleLiquid);
      if (model_type_str == kwds.model_single_liquid)
        model_type = Model::ModelType::SingleLiquid;
      else if (model_type_str == kwds.model_water_oil)
        model_type = Model::ModelType::WaterOil;
      // else if (model_type_str == kwds.model_single_gas)
      //   model_type = Model::ModelType::SingleGas;
      // else if (model_type_str == kwds.model_water_gas)
      //   model_type = Model::ModelType::WaterGas;
      // else if (model_type_str == kwds.model_blackoil)
      //   model_type = Model::ModelType::Blackoil;
      else
        AssertThrow(false, ExcMessage("Wrong entry in " + kwds.model_type));

      {// Permeability & porosity
        const int dim = 3;
        std::vector<double> default_anisotropy{1,1,1};
        Tensor<1,dim> no_anisotropy = Parsers::convert<dim>(default_anisotropy);
        Tensor<1,dim> anisotropy = Parsers::convert<dim>
          (parser.get_double_list(kwds.permeability_anisotropy, ",",
                                  default_anisotropy));
        model.get_permeability =
          get_function(kwds.permeability, anisotropy, parser);

        model.get_porosity =
          get_function(kwds.porosity, no_anisotropy, parser);
      }

      // if (model_type == Model::SingleLiquid ||
      //     model_type == Model::ModelType::WaterOil)
      {
        const auto & tmp = parser.get_matrix(kwds.pvt_water, ";", ",");
        // tmp.print_formatted(std::cout);
        AssertThrow(tmp.n() == model.n_pvt_water_columns,
                    ExcDimensionMismatch(tmp.n(), model.n_pvt_water_columns));
        Interpolation::LookupTable pvt_water_table(tmp);
        model.set_pvt_water(pvt_water_table);
      }

      if (model_type != Model::SingleLiquid &&
          model_type != Model::WaterGas)
      {
        const auto & tmp = parser.get_matrix(kwds.pvt_oil, ";", ",");
        // deadoil: p Bo Co mu_o
        AssertThrow(tmp.n() == model.n_pvt_oil_columns,
                    ExcDimensionMismatch(tmp.n(), model.n_pvt_oil_columns));
        Interpolation::LookupTable pvt_oil_table(tmp);
      }

    } // end equation data

    {  // wells
      parser.enter_subsection(kwds.section_wells);
      assign_wells(kwds.well_parameters, parser);
      assign_schedule(kwds.well_schedule, parser);
    }
    {  // solver
      parser.enter_subsection(kwds.section_solver);
      model.min_time_step =
        parser.get_double(kwds.minimum_time_step, 1e-10);
    }
  } // eom


  Function<3> *
  Reader::get_function(const std::string  &kwd,
                       const Tensor<1,3>  &anisotropy,
                       const SyntaxParser &parser)
  {
    const int dim = 3;
    const auto kwd_list =
      parser.get_str_list(kwd, std::string("\t "));

    std::string entry = kwd_list[0];

    if (entry == "bitmap" && kwd_list.size() == 2)
    { // create bitmap function
      if (verbosity > 0)
        std::cout << "Searching " << kwd_list[1] << std::endl;

      boost::filesystem::path data_file =
        boost::filesystem::path(input_file_name).parent_path() / kwd_list[1];

      return new BitMap::BitMapFunction<dim>(data_file.string(),
                                             anisotropy);
    }
    else if (Parsers::is_number(entry) && kwd_list.size() == 1)
    { // create constant function
      std::vector<double> quantity;
      for (int c=0; c<dim; c++)
        quantity.push_back(boost::lexical_cast<double>(entry)*anisotropy[c]);
      return new ConstantFunction<dim>(quantity);
    }
    else
      AssertThrow(false, ExcNotImplemented());

    return new ConstantFunction<dim>(0);
  } // eom


  void Reader::assign_wells(const std::string  &kwd,
                            const SyntaxParser &parser)
  {
    const auto well_list = parser.get_str_list(kwd, std::string(";"));
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
      const double r = Parsers::convert<double>(well_strs[1]);
      // parse locations
      unsigned int n_loc = (well_strs.size()-2) / 3;
      const int dim = 3;
      std::vector<Point<dim>> locations(n_loc);
      int loc=0;
      for (unsigned int i=2; i<well_strs.size(); i+=dim)
      {
        double x = Parsers::convert<double>(well_strs[i]);
        double y = Parsers::convert<double>(well_strs[i+1]);
        double z = Parsers::convert<double>(well_strs[i+2]);
        locations[loc] = Point<dim>(x,y,z);
        loc++;
      }

      model.add_well(name, r, locations);

    } // end well loop
  }  // eom

  void Reader::assign_schedule(const std::string  &kwd,
                               const SyntaxParser &parser)
  {
    const auto lines = parser.get_str_list(kwd, std::string(";"));
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
      // get skin
      if (entries.size() > 4)
          schedule_entry.control.skin = Parsers::convert<double>(entries[4]);

      model.schedule.add_entry(schedule_entry);
    } // end lines loop
  } // eom
} // end of namespace

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
#include <KeywordReader.hpp>
#include <Model.hpp>


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
    void assign_wells(const std::string   &kwd,
                      const KeywordReader &kwd_reader);
    void assign_schedule(const std::string   &kwd,
                         const KeywordReader &kwd_reader);

    ConditionalOStream                     &pcout;
    Model::Model<3>                        &model;
    int                                    verbosity;
    std::string                            input_text;
  }; //


  Reader::Reader(ConditionalOStream &pcout_,
                 Model::Model<3>    &model_)
    :
    pcout(pcout_),
    model(model_),
    verbosity(0)
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
    read_file(fname);
    Parsers::strip_comments(input_text, "#");
    if (verbosity > 1)
      std::cout << input_text << std::endl;
    Keywords::Keywords kwds;
    KeywordReader kwd_reader(input_text);
    { // Mesh
      kwd_reader.enter_subsection(kwds.section_mesh);
      model.initial_refinement_level =
        kwd_reader.get_int(kwds.global_refinement_steps, 0);
      model.n_adaptive_steps =
        kwd_reader.get_int(kwds.adaptive_refinement_steps, 0);
    }
    {  // equation data
      kwd_reader.enter_subsection(kwds.section_equation_data);
      std::string model_type_str = kwd_reader.get(kwds.model_type);
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
      // std::cout << "type w " << (model_type == Model::ModelType::SingleLiquid) << std::endl;

      if (model_type != Model::ModelType::SingleGas &&
          model_type != Model::ModelType::SingleGasElasticity)
      {
        const double bw = kwd_reader.get_double(kwds.volume_factor_water, 1.0);
        const double muw = kwd_reader.get_double(kwds.viscosity_water, 1e-3);
        const double rhow = kwd_reader.get_double(kwds.density_sc_water, 1e+3);
        const double cw = kwd_reader.get_double(kwds.compressibility_water, 5e-10);
        model.set_viscosity_w(muw);
        model.set_compressibility_w(cw);
        model.set_density_sc_w(rhow);
        model.set_volume_factor_w(bw);
      }

      if (model_type == Model::ModelType::WaterOil ||
          model_type == Model::ModelType::Blackoil ||
          model_type == Model::ModelType::WaterOilElasticity ||
          model_type == Model::ModelType::BlackoilElasticity)
      {
        kwd_reader.get(kwds.volume_factor_oil);
      }

    } // end equation data

    {  // equation data
      kwd_reader.enter_subsection(kwds.section_wells);
      assign_wells(kwds.well_parameters, kwd_reader);
      assign_schedule(kwds.well_schedule, kwd_reader);
    }
  } // eom


  void Reader::assign_wells(const std::string   &kwd,
                            const KeywordReader &kwd_reader)
  {
    const auto well_list = kwd_reader.get_str_list(kwd, std::string(";"));
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

  void Reader::assign_schedule(const std::string   &kwd,
                               const KeywordReader &kwd_reader)
  {
    const auto lines = kwd_reader.get_str_list(kwd, std::string(";"));
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

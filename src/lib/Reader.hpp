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


  } // eom


  // void Reader::assign_wells(const std::string& text)
  // {
  //   /*
  //     Keyword structure is as follows:
  //     string well_name
  //     double radius
  //     comma-semicolon-separated list locations
  //     example
  //     [Well1, 0.1, 1, (1,1; 2,2)], [Well2, 0.1, 1, (1,1; 2,2)]
  //   */
  //   const auto & delim = std::pair<std::string,std::string>("[","]");
  //   std::vector<std::string> split =
  //     Parsers::split_bracket_group(text, delim);

  //   for (auto & item : split)
  //   {
  //     // std::cout << item << std::endl;
  //     const auto & str_params = Parsers::split_ignore_brackets(item);
  //     const std::string well_name = str_params[0];
  //     const double radius = Parsers::convert<double>(str_params[1]);
  //     // separate separate points
  //     std::vector<std::string> point_strs;
  //     std::vector< Point<dim> > locations;
  //     boost::algorithm::split(point_strs, str_params[2], boost::is_any_of(";"));
  //     for (auto & point_str : point_strs)
  //     {
  //       const auto & point = Parsers::parse_string_list<double>(point_str);
  //       AssertThrow(point.size() == dim,
  //                   ExcMessage("Dimensions don't match"));
  //       Point<dim> p;
  //       for (int d=0; d<dim; d++)
  //         p[d] = point[d];
  //       locations.push_back(p);
  //     }

  //     Wellbore::Wellbore<dim> w(locations, radius, mpi_communicator);
  //     this->wells.push_back(w);

  //     // check if well_id is in unique_well_ids and add if not
  //     if (well_ids.empty())
  //       well_ids[well_name] = 0;
  //     else
  //     {
  //       std::map<std::string, int>::iterator
  //         it = well_ids.begin(),
  //         end = well_ids.end();

  //       for (; it!=end; it++)
  //       {
  //         AssertThrow(it->first != well_name, ExcMessage("Duplicates in wells"));
  //       }

  //       const int id = well_ids.size();
  //       well_ids[well_name] = id;
  //     }
  //   }
  // }  // eom

} // end of namespace

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
      unsigned int n_global_steps =
        kwd_reader.get_int(kwds.global_refinement_steps, 0);
      unsigned int n_adaptive_steps =
        kwd_reader.get_int(kwds.adaptive_refinement_steps, 0);
    }
    {  // equation data
      kwd_reader.enter_subsection(kwds.section_equation_data);
      std::string model_type_str = kwd_reader.get(kwds.model_type);
      std::cout << model_type_str << std::endl;
      Model::ModelType model_type;
      if (model_type_str == kwds.model_single_liquid)
        model_type = Model::ModelType::SingleLiquid;
      else if (model_type_str == kwds.model_single_gas)
        model_type = Model::ModelType::SingleGas;
      else if (model_type_str == kwds.model_water_gas)
        model_type = Model::ModelType::WaterGas;
      else if (model_type_str == kwds.model_water_oil)
        model_type = Model::ModelType::WaterOil;
      else if (model_type_str == kwds.model_blackoil)
        model_type = Model::ModelType::Blackoil;
      else
        AssertThrow(false, ExcMessage("Wrong entry in " + kwds.model_type));
      std::cout << "type w " << (model_type == Model::ModelType::SingleLiquid) << std::endl;
    }
    // std::cout << kwd_reader.get_double(kwds.global_refinement_steps) << std::endl;
    // std::cout << kwd_reader.get_double(kwds.global_refinement_steps) << std::endl;
    // section mesh
    // std::string subsection, tmp;
    // subsection = Parsers::find_substring(input_text,
    //                                      "subsection Mesh","subsection");
    // tmp = Parsers::find_substring(subsection, "subsection Mesh","end");
    // std::cout << subsection << std::endl;


  } // eom
} // end of namespace

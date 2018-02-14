#pragma once
#include <deal.II/lac/full_matrix.h>

#include <Parsers.hpp>

namespace Parsers
{
  class SyntaxParser
  {
  public:
    SyntaxParser(std::string &text_);
    void enter_subsection(const std::string& kwd);
    double get_double(const std::string & kwd,
                      const double default_value) const;
    double get_double(const std::string & kwd) const;
    int get_int(const std::string & kwd,
                const int default_value) const;
    int get_int(const std::string & kwd) const;
    std::string get(const std::string &kwd,
                    const std::string default_value) const;
    std::string get(const std::string &kwd) const;
    std::vector<std::string> get_str_list(const std::string &kwd,
                                          const std::string &delimiter) const;
    std::vector<int>
    get_int_list(const std::string  &kwd,
                 const std::string  &delimiter,
                 const unsigned int size=0) const;
    std::vector<double>
    get_double_list(const std::string  &kwd,
                    const std::string  &delimiter,
                    const unsigned int size=0) const;
    std::vector<double>
    get_double_list(const std::string &kwd,
                    const std::string &delimiter,
                    std::vector<double> &default_value) const;
    FullMatrix<double>
    get_matrix(const std::string  &kwd,
               const std::string  &delimiter_col,
               const std::string  &delimiter_row) const;


  private:
    std::string text, active_text;
    // format
    std::string subsection_prefix, subsection_close,
      comment_begin, comment_close, kwd_close;

  };

  SyntaxParser::SyntaxParser(std::string &text_)
    :
    text(text_),
    subsection_prefix("subsection "),
    subsection_close("subsection"),
    comment_begin("#"),
    comment_close("\n"),
    kwd_close("/")
  {}


  void SyntaxParser::enter_subsection(const std::string &kwd)
  {
    try {
      active_text =
        Parsers::find_substring(text,
                                subsection_prefix + kwd,
                                subsection_close);
    }
    catch (std::exception &exc) { // to the end of file
      // this doesn't work for some reason
      // active_text =
      //   Parsers::find_substring(text,
      //                           // subsection_prefix + kwd + "[^$]+");
      //                           // subsection_prefix + kwd + "[^\z]*");
      //                           subsection_prefix + kwd + ".*");

      std::regex re(subsection_prefix + kwd + "[^$]*");
      std::smatch match;
      std::regex_search (text, match, re);
      active_text = text.substr(match.position(0) +
                                subsection_prefix.size() + kwd.size(),
                                text.size());
      AssertThrow(match.size() <= 1,
                  ExcMessage("Found more than one subsection " + kwd));
      // std::cout << "match.size " << match.size() << std::endl;
      // std::cout << "weird section " << active_text << std::endl;
    }

    // std::cout<<active_text<<std::endl;
  } // eom


  int SyntaxParser::get_int(const std::string &kwd) const
  {
    std::string txt = get(kwd);
    return Parsers::convert<int>(txt);
  } // eom


  int SyntaxParser::get_int(const std::string &kwd,
                             const int default_value) const
  {
    try
    {
      get_int(kwd);
    }
    catch (std::exception &exc)
    {
      return default_value;
    }
    return default_value;
  } // eom


  double SyntaxParser::get_double(const std::string &kwd) const
  {
    std::string txt = get(kwd);
    return Parsers::convert<double>(txt);
  } // eom


  double SyntaxParser::get_double(const std::string &kwd,
                                  const double default_value) const
  {
    try
    {
      return get_double(kwd);
    }
    catch (std::exception &exc)
    {
      // std::cout << "Caught something!!11" << std::endl;
      return default_value;
    }
    return default_value;
  } // eom


  std::string SyntaxParser::get(const std::string &kwd) const
  {
    std::string raw_result =
      Parsers::find_substring(active_text, kwd, kwd_close);
    boost::trim(raw_result);
    boost::algorithm::trim_left(raw_result);
    return raw_result;
  }  // eom


  std::string SyntaxParser::get(const std::string &kwd,
                                const std::string default_value) const
  {
    try
    {
      return get(kwd);
    }
    catch (std::exception &exc)
    {
      return default_value;
    }
    return default_value;
  } // eom


  std::vector<std::string>
  SyntaxParser::get_str_list(const std::string &kwd,
                             const std::string &delimiter) const
  {
    std::string raw_result = get(kwd);
    // cut whitespace stuff
    boost::trim_if(raw_result, boost::is_any_of("\t \n"));
    std::vector<std::string> result;
    boost::algorithm::split(result, raw_result,
                            boost::is_any_of(delimiter),
                            boost::token_compress_on);
    // trim and remove empty
    int count=0;
    for (auto & item: result)
    {
      boost::trim(item);
      if (std::all_of(item.begin(),item.end(),isspace))
        result.erase(result.begin()+count);
      count++;
    }

    return result;
  }  // eom



std::vector<int>
SyntaxParser::get_int_list(const std::string  &kwd,
                           const std::string  &delimiter,
                           const unsigned int size) const
{
  const auto & str_list = get_str_list(kwd, delimiter);
  if (size>0)
    AssertThrow(str_list.size() == size,
                ExcDimensionMismatch(str_list.size(), size));
  std::vector<int> int_list(str_list.size());
  for (unsigned int i=0; i<str_list.size(); ++i)
    int_list[i] = convert<int>(str_list[i]);
  return int_list;
} // eom



std::vector<double>
SyntaxParser::get_double_list(const std::string  &kwd,
                              const std::string  &delimiter,
                              const unsigned int size) const
{
  const auto & str_list = get_str_list(kwd, delimiter);
  if (size>0)
    AssertThrow(str_list.size() == size,
                ExcDimensionMismatch(str_list.size(), size));
  std::vector<double> double_list(str_list.size());
  for (unsigned int i=0; i<str_list.size(); ++i)
    double_list[i] = convert<double>(str_list[i]);
  return double_list;
} // eom


  std::vector<double>
  SyntaxParser::get_double_list(const std::string    &kwd,
                                const std::string    &delimiter,
                                std::vector <double> &default_value) const
  {
    try {
      const auto & result =
        get_double_list(kwd, delimiter, default_value.size());
      return result;
    }
    catch (std::exception &exc) {
      return default_value;
    };
  } // eom


  FullMatrix<double>
  SyntaxParser::get_matrix(const std::string  &kwd,
                           const std::string  &delimiter_row,
                           const std::string  &delimiter_col) const
  {
    std::vector<std::string> rows = get_str_list(kwd, delimiter_row);
    const unsigned int n_rows = rows.size();
    std::vector<std::string> tmp;
    boost::algorithm::split(tmp, rows[0],
                            boost::is_any_of(delimiter_col),
                            boost::token_compress_on);
    const unsigned int n_cols = tmp.size();
    FullMatrix<double> result(n_rows, n_cols);
    for (unsigned int i=0; i<n_rows; ++i)
    {
      boost::algorithm::split(tmp, rows[i],
                              boost::is_any_of(delimiter_col),
                              boost::token_compress_on);
      AssertThrow(tmp.size() == n_cols,
                  ExcDimensionMismatch(tmp.size(), n_cols));
      for (unsigned int j=0; j<n_cols; ++j)
      {
        result(i, j) = Parsers::convert<double>(tmp[j]);
      }
    }

    return result;
  } // eom
}

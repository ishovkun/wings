#pragma once
#include <Parsers.hpp>

namespace Parsers
{
  class KeywordReader
  {
  public:
    KeywordReader(std::string &text_);
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

  private:
    std::string text, active_text;
    // format
    std::string subsection_prefix, subsection_close,
      comment_begin, comment_close, kwd_close;

  };

  KeywordReader::KeywordReader(std::string &text_)
    :
    text(text_),
    subsection_prefix("subsection "),
    subsection_close("subsection"),
    comment_begin("#"),
    comment_close("\n"),
    kwd_close("/")
  {}


  void KeywordReader::enter_subsection(const std::string &kwd)
  {
    active_text =
      Parsers::find_substring(text,
                              subsection_prefix + kwd,
                              subsection_close);
    // std::cout<<active_text<<std::endl;
  } // eom


  int KeywordReader::get_int(const std::string &kwd) const
  {
    std::string txt = get(kwd);
    return Parsers::convert<int>(txt);
  } // eom


  int KeywordReader::get_int(const std::string &kwd,
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


  double KeywordReader::get_double(const std::string &kwd) const
  {
    std::string txt = get(kwd);
    return Parsers::convert<double>(txt);
  } // eom


  double KeywordReader::get_double(const std::string &kwd,
                                   const double default_value) const
  {
    try
    {
      get_double(kwd);
    }
    catch (std::exception &exc)
    {
      return default_value;
    }
    return default_value;
  } // eom


  std::string KeywordReader::get(const std::string &kwd) const
  {
    std::string raw_result = Parsers::find_substring(active_text, kwd, kwd_close);
    boost::trim(raw_result);
    return raw_result;
  }  // eom


  std::string KeywordReader::get(const std::string &kwd,
                                 const std::string default_value) const
  {
    try
    {
      return Parsers::find_substring(active_text, kwd, kwd_close);
    }
    catch (std::exception &exc)
    {
      return default_value;
    }
    return default_value;
  } // eom


  std::vector<std::string>
  KeywordReader::get_str_list(const std::string &kwd,
                              const std::string &delimiter) const
  {
    std::string raw_result = get(kwd);
    std::vector<std::string> result;
    boost::algorithm::split(result, raw_result, boost::is_any_of(delimiter));
    return result;
  }  // eom

}

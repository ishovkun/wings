# lib/CMakeLists.txt

ADD_LIBRARY(lib
  PressureSolver.hpp
  DataBase.hpp
  BitMap.hpp
  Parsers.hpp
  Units.cc
  Tensors.hpp
  Keywords.cc
  Wellbore.hpp
  Schedule.cc
)

DEAL_II_SETUP_TARGET(lib)

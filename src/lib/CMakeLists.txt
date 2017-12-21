# lib/CMakeLists.txt

ADD_LIBRARY(wings
  CellValues.hpp
  PressureSolver.hpp
  DataBase.hpp
  BitMap.hpp
  Parsers.hpp
  Units.cc
  Tensors.hpp
  Keywords.cc
  Wellbore.hpp
  Schedule.cc
  DefaultValues.cc
  Math.hpp
)

DEAL_II_SETUP_TARGET(wings)

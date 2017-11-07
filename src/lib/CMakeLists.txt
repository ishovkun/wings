# lib/CMakeLists.txt

ADD_LIBRARY(lib
  PressureSolver.hpp
  DataBase.hpp
  BitMap.hpp
  Parsers.hpp
  Units.cc
  Tensors.hpp
  Keywords.cpp
  Wellbore.hpp
)

DEAL_II_SETUP_TARGET(lib)

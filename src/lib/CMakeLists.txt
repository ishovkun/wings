# lib/CMakeLists.txt

ADD_LIBRARY(wings SHARED
  Simulator.hpp
  Reader.hpp
  SyntaxParser.hpp
  CellValues.hpp
  PressureSolver.hpp
  ExtraFEData.hpp
  SaturationSolverExplicit.hpp
  Model.hpp
  BitMap.hpp
  Parsers.hpp
  Units.h
  Tensors.hpp
  Keywords.h
  Wellbore.hpp
  Schedule.hpp
  DefaultValues.h
  LookupTable.hpp
  Math.hpp
)

DEAL_II_SETUP_TARGET(wings)

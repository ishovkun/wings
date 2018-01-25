# lib/CMakeLists.txt
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/FEFunction)

ADD_LIBRARY(wings SHARED
  Simulator.hpp
  Reader.hpp
  SyntaxParser.hpp
  CellValues.hpp
  PressureSolver.hpp
  RelativePermeability.hpp
  ExtraFEData.hpp
  FEFunction/FEFunction.hpp
  FEFunction/FEFunctionPVT.hpp
  SaturationSolver.hpp
  Math.hpp
  Model.hpp
  BitMap.hpp
  Parsers.hpp
  Units.h
  Tensors.hpp
  Keywords.h
  Schedule.hpp
  DefaultValues.h
  LookupTable.hpp
  Wellbore.hpp
)

DEAL_II_SETUP_TARGET(wings)

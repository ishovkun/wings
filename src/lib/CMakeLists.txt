# lib/CMakeLists.txt
MESSAGE("including library wings")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/FEFunction)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/CellValues)

ADD_LIBRARY(wings SHARED
  Simulator.hpp
  Reader.hpp
  SyntaxParser.hpp
  CellValues/CellValuesBase.hpp
  CellValues/CellValuesSaturation.hpp
  PressureSolver.hpp
  SolverIMPES.hpp
	ElasticSolver.hpp
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
  OutputHelper.hpp
  ScaleOutputVector.hpp
)

DEAL_II_SETUP_TARGET(wings)

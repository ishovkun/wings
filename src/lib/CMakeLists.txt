# lib/CMakeLists.txt
MESSAGE("including library wings")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/FEFunction)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/Equations)

ADD_LIBRARY(wings SHARED
  SyntaxParser.hpp
  Reader.hpp
  Simulator.hpp
  SolverBuilder.hpp
  SolverIMPES.hpp
	AssembleFlowSystem.hpp
	ElasticSolver.hpp
  Equations/FluidEquationsBase.hpp
  Equations/IMPESPressure.hpp
  Equations/IMPESSaturation.hpp
  RelativePermeability.hpp
  ExtraFEData.hpp
  FEFunction/FEFunction.hpp
  FEFunction/FEFunctionPVT.hpp
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

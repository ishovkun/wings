# lib/CMakeLists.txt
MESSAGE("including library wings")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/FEFunction)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/Equations)

ADD_LIBRARY(wings SHARED
  SyntaxParser.hpp
  Reader.hpp
  Simulator.hpp
  SolverBuilder.hpp
	SolutionValues.h
  FluidSolverBase.hpp
  SolverIMPES.hpp
	AssembleFlowSystem.hpp
  SolidSolverBase.hpp
	ElasticSolver.hpp
  Equations/FluidEquationsBase.hpp
  Equations/IMPESPressure.hpp
  Equations/IMPESSaturation.hpp
  RelativePermeability.hpp
  ExtraFEData.hpp
  FEFunction/FEFunction.hpp
  FEFunction/FEFunctionPVT.hpp
  FEFunction/FEFunctionPS.hpp
  Math.hpp
  Model.hpp
  Probe.hpp
  BitMap/BitMapFile.hpp
  BitMap/BitMapFunction.hpp
  BitMap/BitMapTensorFunction.hpp
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

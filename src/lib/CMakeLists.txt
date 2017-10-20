# lib/CMakeLists.txt

ADD_LIBRARY(lib
  PressureSolver.hpp
  DataBase.hpp
  BitMap.hpp
)

DEAL_II_SETUP_TARGET(lib)

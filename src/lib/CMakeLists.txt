# lib/CMakeLists.txt

ADD_LIBRARY(lib
  PressureSolver.hpp
  DataBase.hpp
)

DEAL_II_SETUP_TARGET(lib)

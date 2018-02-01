# benchmarks/CMakeLists.txt

MESSAGE("Including benchmarks")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/buckley_leverett)
ADD_EXECUTABLE(buckley ${CMAKE_CURRENT_SOURCE_DIR}/buckley_leverett/buckley.cc)

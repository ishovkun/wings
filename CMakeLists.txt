
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.5
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


IF(NOT  DEAL_II_WITH_TRILINOS OR NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_WITH_P4EST = ON
One or both of these combinations are OFF in your installation but at least one is required for this tutorial step.
")
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
#SET(CLEAN_UP_FILES *.log *.gmv *.gnuplot *.gpl *.eps *.pov *.vtk *.ucd *.d2 *.vtu *.pvtu)
PROJECT(wings CXX)

# For YouCompleteMe
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# BOOST
set(Boost_USE_STATIC_LIBS OFF)
# set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_NO_SYSTEM_PATHS OFF)
find_package(Boost 1.65.1
  COMPONENTS filesystem iostreams unit_test_framework REQUIRED)
# find_package(Boost REQUIRED COMPONENTS filesystem iostreams)

IF(NOT ${Boost_FOUND})
  MESSAGE(FATAL_ERROR "
Error! Cannot find boost library!
    ")
ENDIF()

set(BOOST_INCLUDE_DIRS $boost_installation_prefix/include)
include_directories(${Boost_INCLUDE_DIRS})

# lib
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/lib)
ADD_SUBDIRECTORY(src/lib)
SET_TARGET_PROPERTIES (
  wings
  PROPERTIES LINKER_LANGUAGE CXX
)

# set(CMAKE_BUILD_TYPE Debug)
# targets
ADD_EXECUTABLE(wings-pressure ${CMAKE_SOURCE_DIR}/src/wings_pressure.cc)
DEAL_II_SETUP_TARGET(wings-pressure)

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

# linking
TARGET_LINK_LIBRARIES(wings-pressure wings ${Boost_LIBRARIES})

# DEAL_II_INVOKE_AUTOPILOT()

ENABLE_TESTING()
ADD_SUBDIRECTORY(test/test_p)  # single pressure uncoupled with rate well
ADD_SUBDIRECTORY(test/test_pr) # single pressure uncoupled with local refinement with bhp well
ADD_SUBDIRECTORY(test/test_prm) # single pressure uncoupled with local refinement with bhp well and MPI
# ADD_SUBDIRECTORY(test)

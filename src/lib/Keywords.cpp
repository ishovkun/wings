#pragma once

#include <string>
/*
  This class holds the strings that designate the names
  of sections and parameters in the input file
 */
namespace Keywords
{
  class Keywords
  {
  public:
    const std::string
    section_mesh = "Mesh",
      mesh_file = "Mesh file",
      global_refinement_steps = "Global refinement steps",
      adaptive_refinement_steps = "Adaptive refinement steps",
      local_refinement_regions = "Local refinement regions",

      section_equation_data = "Equation data" ,
        young_modulus = "Young modulus",
        poisson_ratio = "Poisson ratio",
        volume_factor_water = "Volume factor water",
        volume_factor_oil = "Volume factor oil",
        volume_factor_gas = "Volume factor gas",
        viscosity_water = "Viscosity water",
        viscosity_oil = "Viscosity oil",
        viscosity_gas = "Viscosity gas",
        compressibility_water = "Compressibility water",
        compressibility_oil = "Compressibility oil",
        compressibility_gas = "Compressibility gas",
        permeability = "Permeability",
        permeability_anisotropy = "Permeability anisotropy",

      section_solver = "Solver",
        t_max = "T max",
        time_stepping = "Time stepping",
        minimum_time_step = "Minimum time step",
        fss_tolerance = "FSS tolerance",
        max_fss_steps = "Max FSS steps";

      };
}

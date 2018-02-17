#pragma once

#include <string>
/*
  This class holds the strings that designate the names
  of sections and parameters in the input file
 */
namespace Keywords
{
  /* class Keywords */
  /* { */
  /* public: */
  const std::string
  section_mesh = "Mesh",
    mesh_file = "Mesh file",
    triangulation = "Triangulation",
    file_msh = "msh",
    file_abaqus = "abaqus",
    global_refinement_steps = "Global refinement steps",
    adaptive_refinement_steps = "Adaptive refinement steps",
    local_refinement_regions = "Local refinement regions",

  section_wells = "Well data" ,
    well_parameters = "Wells",
    well_schedule = "Schedule",

  section_equation_data = "Equation data" ,
    model_type = "Fluid model",
    model_single_liquid = "Liquid",
    model_single_gas = "Gas",
    model_water_oil = "DeadOil",
    model_water_gas = "WaterGas",
    model_blackoil = "Blackoil",
    solid_model_type = "Solid model",
    model_compressibility = "Compressibility",
    model_elasticity = "Elasticity",
    unit_system = "Units",
    si_units = "Metric",
    field_units = "Field",
    young_modulus = "Young modulus",
    poisson_ratio = "Poisson ratio",
    biot_coefficient = "Biot coefficient",
    rock_compressibility = "Rock compressibility",
    volume_factor_water = "Volume factor water",
    // volume_factor_oil = "Volume factor oil",
    // volume_factor_gas = "Volume factor gas",
    viscosity_water = "Viscosity water",
    // viscosity_oil = "Viscosity oil",
    // viscosity_gas = "Viscosity gas",
    density_sc_water = "Density water",
    density_sc_oil = "Density oil",
    density_sc_gas = "Density gas",
    compressibility_water = "Compressibility water",
    pvt_oil = "PVT oil",
    pvt_water = "PVT water",
    rel_perm_water = "Rel perm water",
    rel_perm_oil = "Rel perm oil",
    permeability = "Permeability",
    permeability_anisotropy = "Perm anisotropy",
    porosity = "Porosity",

  section_solver = "Solver",
    t_max = "T max",
    time_stepping = "Time stepping",
    minimum_time_step = "Minimum time step",
    fss_tolerance = "FSS tolerance",
    max_fss_steps = "Max FSS steps",
    solid_linear_solver = "Linear solver solid",
    fluid_linear_solver = "Linear solver fluid",
    linear_solver_direct = "Direct",
    linear_solver_cg = "CG",
    linear_solver_gmres = "GMRES",

  section_boundary_conditions = "Boundary conditions",
      solid_dirichlet_labels = "Displacement boundaries",
      solid_dirichlet_components = "Displacement components",
      solid_dirichlet_values = "Displacement values",
      solid_neumann_labels = "Stress boundaries",
      solid_neumann_components = "Stress components",
      solid_neumann_values = "Stress values";


  // Output names
  const std::string
      vtu_dir_name = "vtu",
      vtu_file_prefix = "solution-",
      vtu_file_suffix = "vtu",
      pvtu_file_prefix = "solution-",
      pvtu_file_suffix = "pvtu",
      pvd_file_name = "solution.pvd";

  const std::string
      pressure_vector = "pressure",
      saturation_water_vector = "Sw";

  /* }; */
}

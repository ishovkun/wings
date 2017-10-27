
namespace Units
{
  enum UnitSystem {si_units, field_units};

  class Units
  {
  public:
    // Units();
    void set_system(const UnitSystem&);
    double length() const {return length_constant;}
    double time() const {return time_constant;}
    double pressure() const {return pressure_constant;}
    double viscosity() const {return viscosity_constant;}
    double permeability() const {return permeability_constant;}
    double compressibility() const {return 1.0/stiffness_constant;}
    double gas_rate() const {return gas_rate_constant;}
    double fluid_rate() const {return fluid_rate_constant;}
    double stiffness() const {return stiffness_constant;}
    double transmissibility() const {return transmissibility_constant;}

  private:
    UnitSystem unit_system;
    void compute_quantities();
    double
      length_constant,
      time_constant,
      pressure_constant,
      viscosity_constant,
      fluid_rate_constant,
      gas_rate_constant,
      stiffness_constant,
      permeability_constant,
      transmissibility_constant;
    // conversion constants
    const double
      pounds_per_square_inch = 6894.76,
      centipoise = 1e-3,
      feet = 0.3048,
      day = 60*60*24,
      us_oil_barrel =  0.158987295,
      darcy = 9.869233e-13,
      milidarcy = darcy*1e-2,
      standard_cubic_feet = feet*feet*feet;
  };


  void Units::set_system(const UnitSystem& unit_system_)
  {
    unit_system = unit_system_;
    compute_quantities();
  }  // eom


  void Units::compute_quantities()
  {
    if (unit_system == si_units)
    {
      length_constant = 1;
      time_constant = 1;
      pressure_constant = 1;
      viscosity_constant = 1;
      fluid_rate_constant = 1;
      gas_rate_constant = 1;
      stiffness_constant = 1;
      permeability_constant = 1;
    }
    else if (unit_system == field_units)
    {
      length_constant = feet;
      time_constant = day;
      pressure_constant = pounds_per_square_inch;
      viscosity_constant = centipoise;
      fluid_rate_constant = us_oil_barrel/day;
      gas_rate_constant = standard_cubic_feet/day;
      stiffness_constant = pressure_constant;
      permeability_constant = milidarcy;
    }
    transmissibility_constant =
      permeability_constant*length_constant/viscosity_constant;
  }  // eom
}

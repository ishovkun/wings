#pragma once

namespace Model
{
// using namespace dealii;

class RelativePermeability
{
 public:
  void set_data(
      const double Sw_crit,
      const double So_rw,
      const double k_rw0,
      const double k_ro0,
      const double nw,
      const double no);
  void get_values(const std::vector<double> &saturations,
                  std::vector<double>       &dst) const;

 private:
  double Sw_crit, So_rw, k_rw0, k_ro0, nw, no;
};



void RelativePermeability::set_data(
    const double Sw_crit,
    const double So_rw,
    const double k_rw0,
    const double k_ro0,
    const double nw,
    const double no)
{
  this->Sw_crit = Sw_crit;
  this->So_rw = So_rw;
  this->k_rw0 = k_rw0;
  this->k_ro0 = k_ro0;
  this->nw = nw;
  this->no = no;
} // eom



inline
void RelativePermeability::get_values(const std::vector<double> &saturation,
                                      std::vector<double>       &dst) const
{
  AssertThrow(saturation.size() == 2,
              dealii::ExcDimensionMismatch(saturation.size(), 2));
  AssertThrow(dst.size() == 2,
              dealii::ExcDimensionMismatch(dst.size(), 2));

  const double Sw = saturation[0];

  // dimensionless saturation
  double Sw_d = (Sw - Sw_crit) / (1.0 - Sw_crit - So_rw);
  // clip between 0 and 1
  Sw_d = std::max(0.0, std::min(Sw_d, 1.0));

  const double k_rw = k_rw0 * std::pow(Sw_d, nw);
  const double k_ro = k_ro0 * std::pow(1.0 - Sw_d, no);

  dst[0] = k_rw;
  dst[1] = k_ro;
}  // eom

}  // end namespace

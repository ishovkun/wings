#pragma once

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <deal.II/base/exceptions.h>

namespace Schedule
{
  enum WellControlType {pressure_control, flow_control_total, flow_control_water,
                        flow_control_oil, flow_control_gas};

  const std::map<int, WellControlType> well_control_type_indexing =
    {
      {0, WellControlType::flow_control_total},
      {1, WellControlType::flow_control_water},
      {2, WellControlType::flow_control_oil},
      {3, WellControlType::flow_control_gas},
      {4, WellControlType::pressure_control}
    };

  struct WellControl
  {
    WellControlType type = WellControlType::flow_control_total;
    double value = 0;
    double skin = 0;
  };


  struct ScheduleEntry
  {
    double time;
    // std::string well_name;
    int well_id;
    WellControl control;
  };


  class Schedule
  {
  public:
    // Schedule();
    void add_entry(const ScheduleEntry& entry);
    WellControl get_control(const double time, const int well_id) const;
    // std::vector<WellControl> get_well_controls(const double time);
  private:
    std::vector<double> times;
    std::vector<int> well_ids, unique_well_ids;
    std::vector<WellControl> controls;
  };


  void Schedule::add_entry(const ScheduleEntry& entry)
  {
    /* Add schedule entry to the vectors */
    times.push_back(entry.time);
    well_ids.push_back(entry.well_id);
    controls.push_back(entry.control);

    // check if well_id is in unique_well_ids and add if not
    if (unique_well_ids.empty())
      unique_well_ids.push_back(entry.well_id);
    else
      if (std::find(unique_well_ids.begin(), unique_well_ids.end(), entry.well_id)
          == unique_well_ids.end())
        unique_well_ids.push_back(entry.well_id);
  } // eom


  WellControl Schedule::get_control(const double time, const int well_id) const
  {
    AssertThrow(times.size() > 0, dealii::ExcMessage("Schedule is empty"));

    WellControl control;
    control.value = 0;
    control.type = WellControlType::flow_control_total;

    for (unsigned int i=0; i<times.size(); ++i)
    {
      const double t = this->times[i];
      if (t > time)
        break;
      else
      {
        if (well_id != well_ids[i])
          continue;
        else
          control = controls[i];
      }
    } // end loop in schedule entries

    return control;
  }  // eom

}  // end of namespace

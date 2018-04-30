#pragma once

#include <deal.II/numerics/data_out.h>

// Custom modules
#include <Keywords.h>
#include <DefaultValues.h>

namespace Wings
{


namespace Output

{
using namespace dealii;

template <int dim>
class OutputHelper
{
 public:
  OutputHelper(MPI_Comm &mpi_communicator,
               const parallel::distributed::Triangulation<dim> &triangulation);
  // returns the output directory for the current case
  boost::filesystem::path output_directory();
  /*
   * create directory with the case name and
   * create vtu subdirectory
   */
  void prepare_output_directories();
  void set_case_name(const std::string &case_name);
  void write_output(const double        time,
                    const unsigned int  time_step_number,
                    const DataOut<dim>       &data_out);

 private:
  MPI_Comm                                          & mpi_communicator;
  const parallel::distributed::Triangulation<dim>   & triangulation;
  std::string                                         case_name;
  std::vector< std::pair<double,std::string> >        times_and_names;

};



template<int dim>
OutputHelper<dim>::
OutputHelper(MPI_Comm &mpi_communicator,
             const parallel::distributed::Triangulation<dim> &triangulation)
    :
    mpi_communicator(mpi_communicator),
    triangulation(triangulation)
{}  // end do_something



template<int dim>
void
OutputHelper<dim>::set_case_name(const std::string &case_name)
{
  this->case_name = case_name;
}  // end set_case_name



template<int dim>
boost::filesystem::path
OutputHelper<dim>::output_directory()
{
  boost::filesystem::path path("./" + case_name);
  return path;
}



template<int dim>
void
OutputHelper<dim>::prepare_output_directories()
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    const boost::filesystem::path output_dir = output_directory();

    if (!boost::filesystem::is_directory(output_dir))
    {
      std::cout << "Output folder not found\n"
            << "Creating directory: ";
      if (boost::filesystem::create_directory(output_dir))
        std::cout << "Success" << std::endl;
    }
    else
    { // remove everything from this directory
      std::cout << "Folder exists: cleaning folder: ";
      boost::filesystem::remove_all(output_dir);
      if (boost::filesystem::create_directory(output_dir))
        std::cout << "Success" << std::endl;
    }

    // create directory for vtu's
    const boost::filesystem::path vtu_dir(Keywords::vtu_dir_name);
    // boost::filesystem::path vtu_path("./" + case_name + "/vtu");
    // boost::filesystem::path vtu_path("./" + case_name + "/vtu");
    boost::filesystem::create_directory(output_dir / vtu_dir);
  }  // end mpi==0

} // eom



template<int dim>
void
OutputHelper<dim>::write_output(const double        time,
                                const unsigned int  time_step_number,
                                const DataOut<dim> &data_out)
{
  const std::string output_folder_path = ("./" + case_name + "/");
  const std::string vtu_folder_path =
      (output_folder_path + Keywords::vtu_dir_name + "/");
  const std::string this_vtu_file_name =
      + "/" + Keywords::vtu_file_prefix
      // time step #
      + Utilities::int_to_string(time_step_number,
                                 DefaultValues::n_time_step_digits)
      // process #
      + "." + Utilities::int_to_string(triangulation.locally_owned_subdomain(),
                                       DefaultValues::n_processor_digits)
      // extension (.vtu)
      + "." + Keywords::vtu_file_suffix;

  std::ofstream vtu_file(( vtu_folder_path + this_vtu_file_name ).c_str());
  data_out.write_vtu(vtu_file);

  // Write master pvtu and pvd files
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    // Write master pvtu file (combines multiple pvtu's)
    std::vector<std::string> all_vtu_files;
    // loop through number of processors to compose vtu names
    for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
    {
      const std::string vtu_file_name =
          Keywords::vtu_file_prefix
          // time step #
          + Utilities::int_to_string(time_step_number,
                                     DefaultValues::n_time_step_digits)
          // process #
          + "." + Utilities::int_to_string (i, DefaultValues::n_processor_digits)
          // extension (.vtu)
          + "." + Keywords::vtu_file_suffix;

      all_vtu_files.push_back(vtu_file_name);
    } // end all vtu filename loop

    const std::string pvtu_filename =
        Keywords::pvtu_file_prefix
        + Utilities::int_to_string(time_step_number,
                                   DefaultValues::n_time_step_digits)
        + "." + Keywords::pvtu_file_suffix;

    std::ofstream pvtu_file(( vtu_folder_path + pvtu_filename ).c_str());
    data_out.write_pvtu_record(pvtu_file, all_vtu_files);

    // write master pvd file (for real time in paraview)
    const std::string pvtu_full_name = Keywords::vtu_dir_name + "/" + pvtu_filename;
    times_and_names.push_back(std::make_pair(time, pvtu_full_name));
    std::ofstream pvd_file(output_folder_path + Keywords::pvd_file_name);
    DataOutBase::write_pvd_record(pvd_file, times_and_names);
  }  // end if proc 0

}  // end write_output

}  // end of namespace


}  // end Wings

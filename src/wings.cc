#include <deal.II/base/mpi.h>

#include <Parsers.hpp>
#include <Reader.hpp>
#include <Model.hpp>
#include <Simulator.hpp>

/*
 * This code reads the input file,
 * creates and fills the model,
 * creates the simulator instance and runs it
 */

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    dealii::deallog.depth_console(0);
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::string input_file_name = Wings::Parsers::parse_command_line(argc, argv);

    MPI_Comm mpi_communicator(MPI_COMM_WORLD);

    dealii::ConditionalOStream
        pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

    const int dim = 3;
    Wings::Model::Model<dim> model(mpi_communicator, pcout);

    Wings::Parsers::Reader reader(pcout, model);
    reader.read_input(input_file_name, /* verbosity= */0);


    switch (model.n_phases())
    {
      case 1:
      {
        Wings::Simulator<dim, 1> simulator(model, mpi_communicator, pcout);
        simulator.run();
        break;
      }
      // case 2:
      //   {
      //     Wings::Simulator<dim, 2> simulator(model, mpi_communicator, pcout);
      //     simulator.run();
      //     break;
      //   }
      // case 3:
      //   {
      //     Wings::Simulator<dim, 3> simulator(model, mpi_communicator, pcout);
      //     simulator.run();
      //     break;
      //   }
    } // end case n_phases

    return 0;
  } // end try

  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

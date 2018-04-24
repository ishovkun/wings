#include <Simulator.hpp>
#include <Parsers.hpp>
#include <Reader.hpp>
#include <Model.hpp>
/*
 */

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    dealii::deallog.depth_console(0);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::string input_file_name = Parsers::parse_command_line(argc, argv);

    ConditionalOStream
    pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

    MPI_Comm mpi_communicator(MPI_COMM_WORLD);

    Model::Model<dim> model(mpi_communicator, pcout);

    Parsers::Reader reader(pcout, model);
    reader.read_input(input_file, /* verbosity= */0);

    const int dim = 3;

    switch (model.n_phases())
    {

      case 1:
      {
        Wings::Simulator<dim, 1> simulator(model, pcout);
        simulator.run();
        break;
      }
      case 2:
        {
          Wings::Simulator<dim, 2> simulator(model, pcout);
          simulator.run();
          break;
        }
      case 3:
        {
          Wings::Simulator<dim, 3> simulator(model, pcout);
          simulator.run();
          break;
        }
      else
    }

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

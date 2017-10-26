# Wings
**Wings** is a blackoil reservoir simulator coupled with geomechanics.
It is a work in progress and currently has pretty much nothing.

## Main idea
The main insentive for this software is to couple FEM-based mechanical equations
with FVM-based fluid flow within the framework of **deal.ii** library (which is done
rarely if ever).

## Usage
This code requires dealii-8.5 to work.
Furthermore, deal-ii must be compiled and linked with Trilinos and p4est.
### Building Trilinos
I took the instructions for there:
https://www.dealii.org/8.5.0/external-libs/trilinos.html

~~~~
git clone https://github.com/trilinos/Trilinos
cd Trilinos
mkdir build; cd build
cmake                                                \
    -DTrilinos_ENABLE_Amesos=ON                      \
    -DTrilinos_ENABLE_Epetra=ON                      \
    -DTrilinos_ENABLE_Ifpack=ON                      \
    -DTrilinos_ENABLE_AztecOO=ON                     \
    -DTrilinos_ENABLE_Sacado=ON                      \
    -DTrilinos_ENABLE_Teuchos=ON                     \
    -DTrilinos_ENABLE_MueLu=ON                       \
    -DTrilinos_ENABLE_ML=ON                          \
    -DTrilinos_VERBOSE_CONFIGURE=OFF                 \
    -DTPL_ENABLE_MPI=ON                              \
    -DBUILD_SHARED_LIBS=ON                           \
    -DCMAKE_VERBOSE_MAKEFILE=OFF                     \
    -DCMAKE_BUILD_TYPE=RELEASE                       \
    -DCMAKE_INSTALL_PREFIX:PATH=$HOME/share/trilinos \
    ../

make
make install
~~~~

### Building boost (may be optional)
The default boost libraries from the Ubuntu 16.04 repositories were too old,
so that cmake preferred to used the boost version bundled with deal.ii.
In order to overcome this, I had to build boost as well
The latest sources are available at
http://www.boost.org/users/download/
I used boost-1.65.
To build boost I used the following commands:
~~~~
./bootstrap.sh --prefix=$HOME/share
./b2 --prefix=$HOME/share/boost-1.65 install
~~~~

### Building deal.ii
~~~~
git clone https://github.com/dealii/dealii
cd dealii
mkdir build; cd build

cmake -DCMAKE_INSTALL_PREFIX=$HOME/share/dealii \
      -DDEAL_II_WITH_MPI=ON \
      -DDEAL_II_WITH_TRILINOS=ON \
      -DDEAL_II_WITH_P4EST=ON \
      -DTRILINOS_DIR=$HOME/share/trilinos \
      -DP4EST_DIR=/path/to/p4est \
      ..
make
make install
~~~~

### Compiling Wings
After all the above it should be easy (yes, in-source builds are supported :-) ):
~~~~
git clone https://github.com/ishovkun/wings
cd wings
cmake -DDEAL_II_DIR=$HOME/share/dealii .
make
./wings-pressure test/input/test-3x3-homog.prm
~~~~
In case if you had to build boost as well, you need to specify the BOOST_ROOT
as well:
~~~~
cmake -DDEAL_II_DIR=$HOME/share/dealii -DBOOST_ROOT=$HOME/share/boost-1.65.
~~~~

# Wings
**Wings** is a coupled parallel geomechanics-blackoil reservoir simulator with adaptive mesh refinement.
It is a work in progress and currently has only unit tests and some benchmakrs (comparisons with analytical solutions).
To learn more about **Wings**, check out the [Wiki](www.github.com/ishovkun/wings/wiki).

Developers and testers are welcome! The build instructions can be found in the
[Build instructions Wiki](https://github.com/ishovkun/wings/wiki/Build-instructions).
Some [input file examples](https://github.com/ishovkun/wings/tree/master/test/data) can be found in the code repo.

## Main idea
The main insentive for this software is to couple FEM-based mechanical equations
with FVM-based fluid flow within the framework of [deal.ii](http://www.dealii.org) library (which is done
rarely if ever).

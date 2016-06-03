#ifndef __homework_4_heat_h
#define __homework_4_heat_h

#include <mpi.h>

/*
  heat_serial

  A serial version of solving the periodic heat equation on a uniform grid. Use
  for comparison with the MPI distributed version.

  The forward Euler method for numerical solving PDEs is unstable for large
  (even only moderately) values of `dt`. Make sure `dt` is small. Examples are
  given in the Python code that calls this function.

  Parameters
  ----------
  u : double*
      Input data array. Represents the heat / temperature across a 1d rod.
      Assumes that the data points are equally spaced in the space domain.
  Nx : size_t
      The length of `u`.
  dt : double
      The size of the time step to take with each iteration.
  Nt : size_t
      Number of time-steps to perform.

  Returns
  -------
  None
      The array `u` is modified in place.

*/

/*
  heat_parallel

  A parallel version of solving the periodic heat equation on a uniform grid. Parallelization of 
  the 1-D heat transfer is done via MPI. Four ghost cells are used in order to send and recieve
  data from the left and right boundaries in order to simulate the periodic boundary conditions. 



  Parameters
  ----------
  uk : double*
      Input data array. Represents the heat / temperature across a 1d rod.
      Assumes that the data points are equally spaced in the space domain.
  Nx : size_t
      The length of `u`.
  dt : double
      The size of the time step to take with each iteration.
  Nt : size_t
      Number of time-steps to perform.
  MPI_comm: comm
      Used to pass data via MPI

  Returns
  -------
  None
      The array `u` is modified in place.

*/
void heat_serial(double* u, double dx, size_t Nx, double dt, size_t Nt);


void heat_parallel(double* uk, double dx, size_t Nx, double dt, size_t Nt,
                   MPI_Comm comm);

#endif

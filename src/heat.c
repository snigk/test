#include <stdlib.h>
#include <mpi.h>

/*
  array_copy

  Copies the contents of one array to another. Note that this is "stronger" than
  just having ```toarray` point to the same location as `fromarray`.

  Parameters
  ----------
  toarray : double*
      Array to copy into.
  fromarray : double*
      Array to copy the contents from into `toarray`.
  length : size_t

  Returns
  -------
  None
      `toarray` is modified in-place.
*/
void array_copy(double* toarray, double* fromarray, size_t length)
{
  for (size_t i=0; i<length; ++i)
    toarray[i] = fromarray[i];
}

/*
  heat_serial

  See include/heat.h for documentation.
*/
void heat_serial(double* u, double dx, size_t Nx, double dt, size_t Nt)
{
  // allocate temporary space for the time stepping routine
  double* ut = (double*) malloc(Nx * sizeof(double));
  double* utp1 = (double*) malloc(Nx * sizeof(double));
  double* temp;

  // copy contents of `u` to `ut`
  array_copy(ut, u, Nx);

  // determine the numerical diffusion coefficient
  double nu = dt/(dx*dx);

  // time step `Nt` times with step size `dt`
  for (size_t step=0; step<Nt; ++step)
    {
      // update using Forward Euler
      for (size_t i=1; i<(Nx-1); ++i)
        utp1[i] = ut[i] + nu*(ut[i-1] - 2*ut[i] + ut[i+1]);

      // handle periodic boundary conditions: that is, assume a "wrap-around"
      // from the left edge of the boundary to the right edge
      utp1[0] = ut[0] + nu*(ut[Nx-1] - 2*ut[0] + ut[1]);
      utp1[Nx-1] = ut[Nx-1] + nu*(ut[Nx-2] - 2*ut[Nx-1] + ut[0]);

      // iterate: here we use some pointer arithmetic trickery to make `ut`
      // point to the newly-generated data and make `utp1` point to the old data
      // (which we will write over in the next iteration)
      //
      // again: `ut`, `utp1`, and `temp` are just pointers to location in
      // memory. this is a really cheap way to "swap" array data. though,
      // depending on what you're doing you need to be careful.
      temp = ut;
      ut = utp1;
      utp1 = temp;
    }

  // copy the results (located at `ut`) to `u`. that is, the location in memory
  // pointed to by `u` now contains the solution data.
  array_copy(u, ut, Nx);

  // free allocations (make sure the contents of `ut` are copied to `u` before
  // freeing these allocations!)
  free(ut);
  free(utp1);
}

/*
  heat_parallel

  SEE BELOW ON WHERE YOU NEED TO WRITE YOUR C / MPI CODE.
 */
void heat_parallel(double* uk, double dx, size_t Nx, double dt, size_t Nt,
                   MPI_Comm comm)
{
  // get information about the MPI environment and create spaces for Status and
  // Request information (if needed)
  MPI_Status stat;
  MPI_Request req;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // allocate temporary space for this process's chunk of the heat equation
  // calculation.
  double* ukt = (double*) malloc(Nx * sizeof(double));
  double* uktp1 = (double*) malloc(Nx * sizeof(double));
  double* temp;

  // copy contents of the input initial chunk data to ut for iteration
  array_copy(ukt, uk, Nx);

  // set the numerical diffusion coefficient
  double nu = dt/(dx*dx);

  /*
	START IMPLEMENT YOUR SOLUTON HERE ZONE
  */
  // keeping track of which process is left and right of the current one,
  // which is as god intended
//  int left_process(rank-1 + size) % size;
//  int right_process(rank+1) % size;
// ok this part will basically send your rank -1 to your rank
    if ( 0 < rank )
    {
      MPI_Send ( &uk[1], 1, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD );
    }
// gotta get rank + 1 from size - 1
    if ( rank < size-1 )
    {
      MPI_Recv ( &uk[Nx+1], 1,  MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &stat );
    }
    if ( rank < size-1 )
    {
      MPI_Send ( &uk[Nx], 1, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD );
    }
    if ( 0 < rank )
    {
      MPI_Recv ( &uk[0], 1, MPI_DOUBLE, rank-1, 2, MPI_COMM_WORLD, &stat );
    }
/*
  // basically you just want to make ghost cells that hold onto the ut calculation within each block at the right of the left
  // and then do a wraparound that feeds it?
  */
      for (size_t step=0; step<Nt; ++step)
        {
          uktp1[step] = ukt[step] + nu*(ukt[step-1] - 2*ukt[step] + ukt[step+1]);

          /*
            Your tasks:

            1) Communicate necessary boundary data

            2) Solve the heat equation using the Forward Euler method in this part
            of the domain.

           */
        }
    // if we actually had boundaries we'd probably need to recompute them
  /*
	END IMPLEMENT YOUR SOLUTION HERE ZONE
   */
  // copy contents of solution, stored in `ut` to the input chunk `uk`. that is,
  // the location pointed to by `uk` now contains the data after iteration.
  array_copy(uk, ukt, Nx);

  // free temporary allocations
  free(ukt);
  free(uktp1);
}

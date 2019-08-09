#include "eigens.h"
#include <mkl.h>
#include <string.h>
#include <stdio.h>

/*****************************************************/
/*
  @function ComputeEigens
    Compute the eigenvalues and eigenvectors of a given matrix.

  @param matrixIn
    Pointer to the input matrix (n x n).

  @param n
    Dimensions of the matrix.

  @param eigenvaluesOut
    Pointer to the output eigenvalues.

  @param eigenvectorOut
    The number of nearest neighbors

  @return
    Error code of function.
*/
/*****************************************************/
int ComputeEigens(void* matrixIn, size_t n, void** eigenvaluesOut, void** eigenvectorOut)
{
  // data size of matrix
  size_t byteCount = n * n * sizeof(float);
  
  // Initialize output with input matrix
  memcpy_s(*eigenvectorOut, byteCount, matrixIn, byteCount);

  // Use lapack function to retrieve eigenvalues and eigenvectors
  MKL_INT res = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U', (MKL_INT)n, (float*)*eigenvectorOut, (MKL_INT)n, (float*)*eigenvaluesOut);
  
  return res;
}

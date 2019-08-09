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

void SelectEigenVector(float* inEigenvector, int d, int n, float* outEigenvector, int k)
{
  size_t byteSize = d * k * sizeof(float);
  float* tmp = (float*)malloc(byteSize);
  memcpy_s(tmp, byteSize, inEigenvector, byteSize);

  int l = 0;
  for (int i = 0; i < d; ++i)
  {
    for (int j = 0; j < k; ++j)
      outEigenvector[l++] = tmp[j * d + i];
  }

  free(tmp);
}

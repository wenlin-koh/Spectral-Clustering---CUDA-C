#include "kmeans.h"
#include <stdlib.h>
#include <string>
#include <mkl.h>

/*****************************************************/
/*
  @function Cluster
    Perform k-means clustering on input.

  @param dIn
    Pointer to the datasets. (d x n)

  @param groupOut
    Pointer to the output group id. (d x 1)

  @param k
    The number of cluster.
*/
/*****************************************************/
void Cluster(float* dIn, int n, int d, int* groupOut, int k)
{
  // Temporary vector storage
  float *a, *b, *c;
  const size_t dataSize = d * sizeof(float);
  a = (float*)malloc(dataSize);
  b = (float*)malloc(dataSize);
  c = (float*)malloc(dataSize);

  // Set up initial mean using first k-th data points in the dataset
  const size_t meanSize = k * d * sizeof(float);
  float* mean = (float*)malloc(meanSize);
  
  // Storage for updating means
  float *sum = (float*)malloc(meanSize);
  int *count = (int*)malloc(k * sizeof(int));

  cblas_scopy(k * d, dIn, 1, mean, 1);

  // Start k-means algorithm
  float err = 0.0f;
  int NumOfIteration = 2000;
  while (--NumOfIteration)
  {
    // Group data
    // For each data, group data to the shortest distance to a mean.
    for (int i = 0; i < n; ++i)
    {
      // Copy data[i] into a
      cblas_scopy(d, &dIn[i * d], 1, a, 1);

      float shortestDist = INFINITY;
      groupOut[i] = 0;
      
      // Check distances to each mean
      for (int ck = 0; ck < k; ++ck)
      {
        // Copy mean[ck] into b
        cblas_scopy(d, &mean[ck * d], 1, b, 1);
        
        // Subtract a - b to get vector c
        vsSub(d, a, b, c);

        // Compute euclidean distance between a and b
        float dist = cblas_snrm2(d, c, 1);

        // Update the shortest group to groupOut
        if (dist < shortestDist)
        {
          shortestDist = dist;
          groupOut[i] = ck;
        }
      }
    }

    // Update means
    memset(sum, 0, meanSize);
    memset(count, 0, k * sizeof(int));

    // For each data set compute the new mean after grouping
    for (int i = 0; i < n; ++i)
    {
      // Copy data[i] into a
      cblas_scopy(d, &dIn[i * d], 1, a, 1);

      // Accumulate all cluster's data
      vsAdd(d, a, &sum[groupOut[i] * d], &sum[groupOut[i] * d]);

      // Counter to count the size of each group
      ++count[groupOut[i]];
    }    

    for (int ck = 0; ck < k; ++ck)
    {
      // Compute new mean by averaging the sum of dataset in each cluster
      if(count[ck] > 0)
        cblas_sscal(d, 1 / (float)count[ck], &sum[ck * d], 1);

      // Update the mean values
      cblas_scopy(d, &sum[ck * d], 1, &mean[ck * d], 1);
    }
  }

  free(a);
  free(b);
  free(c);
  
  free(mean);
  free(sum);
  free(count);
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

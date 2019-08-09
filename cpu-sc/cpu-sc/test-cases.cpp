#include "test-cases.h"
#include "eigens.h"
#include "laplacian.h"
#include "load.h"
#include "kmeans.h"
#include <mkl.h>
#include <string>
#include <stdlib.h>
#include <stdio.h>

void printMatrix(const char* title, float* matrix, int row, int col)
{
  int i, j;
  printf("\n %s\n", title);
  for (i = 0; i < row; ++i)
  {
    for (j = 0; j < col; ++j)
      printf(" %6.2f", matrix[i + j * row]);
    printf("\n");
  }
}

int testEigen1()
{
  printf("=====================================");
  printf("\n testEigen1()\n");

  // Dimensions
  const int n = 5;

  // Allocate to store results
  void* eigenvalues = malloc(n * sizeof(float));
  void* eigenvectors = malloc(n * n * sizeof(float));

  // Input test
  float input_matrix[n * n] = {
      6.39f,  0.00f,  0.00f,  0.00f,  0.00f,
      0.13f,  8.37f,  0.00f,  0.00f,  0.00f,
     -8.23f, -4.46f, -9.58f,  0.00f,  0.00f,
      5.71f, -6.10f, -9.25f,  3.72f,  0.00f,
     -3.18f,  7.21f, -7.42f,  8.54f,  2.51f
  };

  /* Solve eigenproblem */
  int res = ComputeEigens(input_matrix, n, &eigenvalues, &eigenvectors);

  printMatrix("Input", input_matrix, n, n);
  printMatrix("Eigenvalues", (float*)eigenvalues, 1, n);
  printMatrix("Eigenvectors", (float*)eigenvectors, n, n);
  /*
    Eigenvalues
    - 17.44 - 11.96   6.72  14.25  19.84

    Eigenvectors(stored columnwise)
    -0.26  0.31 -0.74  0.33  0.42
    -0.17 -0.39 -0.38 -0.80  0.16
    -0.89  0.04  0.09  0.03 -0.45
    -0.29 -0.59  0.34  0.31  0.60
    -0.19  0.63  0.44 -0.38  0.48
  */

  // Release memory
  free(eigenvalues);
  free(eigenvectors);

  printf("=====================================\n");
  return res;
}

int testLaplacian1()
{
  printf("=====================================");
  printf("\n testLaplacian1()\n");
	float adj[] =
	{
		0.f, 1.f, 0.f, 1.f,
		1.f, 0.f, 1.f, 1.f,
		0.f, 1.f, 0.f, 1.f,
		1.f, 1.f, 1.f, 0.f,
	};

	float l[16]{};

	Laplacian(l, adj, 4);
	printMatrix("Adjacency", adj, 4, 4);

	printMatrix("Laplacian", l, 4, 4);

  printf("=====================================\n");

	return 0;
}

int testLoadData1()
{
	int r, c;
	auto data = FromFile("Data/Test1.txt", r, c);
	printMatrix("Data Matrix 1", data, r, c);
	free((void*)data);
	return 0;
}

int testLoadData2()
{
	int r, c;
	auto data = FromFile("Data/Test2.txt", r, c);
	printMatrix("Data Matrix 2", data, r, c);
	free((void*)data);
	return 0;
}

int testLoadData3()
{
	int r, c;
	auto data = FromFile("Data/Test3.txt", r, c);
	printMatrix("Data Matrix 3", data, r, c);
	free((void*)data);
	return 0;
}

int testKMeans1()
{
  printf("=====================================");
  printf("\n testKMeans1()\n");

  // Dimensions
  const int n = 20;
  const int d = 2;

  // input dataset (20, 2)
  float data[40] = { 
    -84.5152f, 19.1264f,
    -76.6465f, 20.2474f,
    -76.6615f, 20.342f,
    -76.6415f, 20.2342f,
    -76.6472f, 20.2203f,
    -76.6409f, 20.2693f,
    -64.9945f, 9.5567f,
    -83.1603f, 23.0362f,
    -82.5861f, 23.6002f,
    -80.3366f, 22.236f,
    -92.0933f, 21.0152f,
    -76.1797f, 20.2258f,
    -83.1856f, 28.1131f,
    -80.8078f, 22.6431f,
    -99.3341f, 27.3178f,
    -99.4605f, 27.4186f,
    -83.198f , 28.0765f,
    -80.9032f, 22.7317f,
    -83.2088f, 28.0703f,
    -108.714f, 29.7114f,
  };


  int* groupData = (int*)malloc(n * sizeof(int));
  float err = 0.0f;
  Cluster(data, n, d, groupData, 2);

  // Error is 4.976
  printf("Error : %6.3f\n", err);

  printf("=====================================\n");
  return 0;
}

int testGroup()
{
  // Dimensions
  const int d = 20;
  const int n = 2;
  const int k = 2;
  
  float *a, *b, *c;
  const size_t dataSize = n * sizeof(float);
  a = (float*)malloc(dataSize);
  b = (float*)malloc(dataSize);
  c = (float*)malloc(dataSize);

  // Set up initial mean using first k-th data points in the dataset
  const size_t meanSize = k * n * sizeof(float);
  float* mean = (float*)malloc(meanSize);

  // input dataset (20, 2)
  float data[40] = {
    -84.5152f, 19.1264f,
    -76.6465f, 20.2474f,
    -76.6615f, 20.342f,
    -76.6415f, 20.2342f,
    -76.6472f, 20.2203f,
    -76.6409f, 20.2693f,
    -64.9945f, 9.5567f,
    -83.1603f, 23.0362f,
    -82.5861f, 23.6002f,
    -80.3366f, 22.236f,
    -92.0933f, 21.0152f,
    -76.1797f, 20.2258f,
    -83.1856f, 28.1131f,
    -80.8078f, 22.6431f,
    -99.3341f, 27.3178f,
    -99.4605f, 27.4186f,
    -83.198f , 28.0765f,
    -80.9032f, 22.7317f,
    -83.2088f, 28.0703f,
    -108.714f, 29.7114f,
  };

  cblas_scopy(k * n, data, 1, mean, 1);
  printMatrix("Mean", mean, k, n);
  
  int* groupData = (int*)malloc(d * sizeof(int));
  float err = 0.0f;

  // Group data
  // For each data, group data to the shortest distance to a mean.
  for (int i = 0; i < d; ++i)
  {
    // Copy data[i] into a
    cblas_scopy(n, &data[i * n], 1, a, 1);

    float shortestDist = INFINITY;
    groupData[i] = 0;

    // Check distances to each mean
    for (int ck = 0; ck < k; ++ck)
    {
      // Copy mean[ck] into b
      cblas_scopy(n, &mean[ck * n], 1, b, 1);

      // Subtract a - b to get vector c
      vsSub(n, a, b, c);

      // Compute euclidean distance between a and b
      float dist = cblas_snrm2(n, c, 1);

      // Update the shortest group to groupOut
      if (dist < shortestDist)
      {
        shortestDist = dist;
        groupData[i] = ck;
      }
    }
  }

  // Storage for updating means
  float *sum = (float*)malloc(meanSize);
  int *count = (int*)malloc(k * sizeof(int));

  // Update means
  memset(sum, 0, meanSize);
  memset(count, 0, k * sizeof(int));

  // For each data set compute the new mean after grouping
  for (int i = 0; i < d; ++i)
  {
    // Copy data[i] into a
    cblas_scopy(n, &data[i * n], 1, a, 1);

    // Accumulate all cluster's data
    vsAdd(n, a, &sum[groupData[i] * n], &sum[groupData[i] * n]);

    // Counter to count the size of each group
    ++count[groupData[i]];
  }

  for (int ck = 0; ck < k; ++ck)
  {
    // Compute new mean by averaging the sum of dataset in each cluster
    cblas_sscal(n, 1 / (float)count[ck], &sum[ck * n], 1);

    // Update the mean values
    cblas_scopy(n, &sum[ck * n], 1, &mean[ck * n], 1);
  }

  // Compute error to check for convergence
  float currentErr = 0.0f;
  for (int i = 0; i < d; ++i)
  {
    cblas_scopy(n, &data[i * n], 1, a, 1);
    cblas_scopy(n, &mean[groupData[i] * n], 1, b, 1);
    vsSub(n, a, b, c);
    currentErr += cblas_snrm2(n, c, 1);
  }

  currentErr *= 1.0f / d;

  for (int i = 0; i < d; ++i)
    printf("%2d : %d\n", i, groupData[i]);

  printMatrix("Mean", mean, k, n);
  printf("Error : %6.2f\n", currentErr);
  
  free(a);
  free(b);
  free(c);
  free(mean);

  printf("=====================================\n");
  return 0;
}

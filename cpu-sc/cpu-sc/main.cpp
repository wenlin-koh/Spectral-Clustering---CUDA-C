#include <mkl.h>
#include <iostream>
#include "cpu-sc.h"
#include "test-cases.h"
#include <fstream>
#include <chrono>


const char* files[] = {
  "../../data/Moons_500.txt",     // 0
  "../../data/Circles_500.txt",   // 1
  "../../data/Circles_5000.txt",  // 2
  "../../data/Circles_10000.txt", // 3
};

int main(int argc, char** argv)
{
  const char* fileName;
  if (argc < 4)
  {
    if(argc < 3)
    {
      printf("Usage: .\\app {knn-k value} {kmeans-k value} {fileName} \n\n");
      return 0;
    }
    else
      fileName = files[0];
  }
  else
  {
    fileName = argv[3];
  }
  std::chrono::high_resolution_clock clock;
  
  int knnk = atoi(argv[1]);
  int kmeansk = atoi(argv[2]);

  int r, c;
  float* data = FromFile(fileName, r, c);
  auto timeElapsed = 0.0f;

  // Compute Laplacian
  auto start = clock.now();
  float* adjMatrix = (float*)malloc(c * c * sizeof(float));
  memset(adjMatrix, 0, c * c * sizeof(float));
  
  float* laplacianMatrix = (float*)malloc(c * c * sizeof(float));
  memset(laplacianMatrix, 0, c * c * sizeof(float));

  Adjacency(adjMatrix, data, c, r, knnk);
  Laplacian(laplacianMatrix, adjMatrix, c);
  auto end = clock.now();
  auto diff = (float)(1.0e-3 * (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
  timeElapsed += diff;
  printf("%.5f,", diff);

  // Compute Eigenvalues and Eigenvectors
  start = clock.now();

  void* eigenvalues = malloc(c * sizeof(float));
  memset(eigenvalues, 0, c * sizeof(float));
  
  void* eigenvectors = malloc(c * c * sizeof(float));
  memset(eigenvectors, 0, c * c * sizeof(float));
  
  void* eigenvectorsTmp = malloc(c * kmeansk * sizeof(float));

  ComputeEigens((void*)laplacianMatrix, c, &eigenvalues, &eigenvectors);
  SelectEigenVector((float*)eigenvectors, c, c, (float*)eigenvectorsTmp, kmeansk);

  end = clock.now();
  diff = (float)(1.0e-3 * (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
  timeElapsed += diff;
  printf("%.5f,", diff);

  // Clustering
  start = clock.now();

  int* groupOut = (int*)malloc(c * sizeof(int));
  memset(groupOut, 0, c * sizeof(int));

  Cluster((float*)eigenvectorsTmp, c, kmeansk, groupOut, kmeansk);

  end = clock.now();
  diff = (float)(1.0e-3 * (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()); 
  timeElapsed += diff;
  printf("%.5f,", diff);
  printf("%.5f", timeElapsed);

  // Output to file
  auto of = std::ofstream{ "../../scripts/LabeledPointsCPU.txt" };
  for (auto i = 0; i < c; ++i)
  {
    for (auto j = 0; j < r; ++j)
      of << data[i * r + j] << ",";
    of << groupOut[i] << "\n";
  }

  free(groupOut);
  free(eigenvectorsTmp);
  free(eigenvectors);
  free(eigenvalues);
  free(laplacianMatrix);
  free(adjMatrix);
  free((void*)data);
  return 0;
}
/* Start Header
***************************************************************** /
/*!
\file main.cpp
\author Low Jin Liang Aaron, aaron.low, 390001116
\par aaron.low@digipen.edu
\date 20/5/2019
\brief
Copyright (C) 2019 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <fstream>

const char* files[] = {
  "../../data/Moons_500.txt",     // 0
  "../../data/Circles_500.txt",   // 1
  "../../data/Circles_5000.txt",  // 2
  "../../data/Circles_10000.txt", // 3
};

//add into Project/Properties/CUDA C/C++ Additional Include Directories
//C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.2\common\inc;
// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include <stdint.h>

#include "gpu-sc.h"

/*
StopWatchInterface* hTimer = NULL;
sdkCreateTimer(&hTimer);
sdkResetTimer(&hTimer);
sdkStartTimer(&hTimer);
sdkStopTimer(&hTimer);
auto dAvgSecs = (float)(1.0e-3 * (double)sdkGetTimerValue(&hTimer));
printf("CPU version time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)(count * sizeof(float))  * 1.0e-6) / dAvgSecs);
printf("CPU version, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes\n",
(1.0e-6 * (double)(count * sizeof(float)) / dAvgSecs), dAvgSecs, (unsigned int)(count * sizeof(float)));
*/

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

int main(int argc, char **argv)
{
  StopWatchInterface* hTimer = NULL;
  sdkCreateTimer(&hTimer);

  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;

  int dev = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
  // printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n", deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

  const char* fileName;
  if (argc < 4)
  {
    if (argc < 3)
    {
      printf("Usage: .\\app {knn-k value} {kmeans-k value} {fileName} \n");
      return 0;
    }
    else
      fileName = files[0];
  }
  else
  {
    fileName = argv[3];
  }

  int knnk = atoi(argv[1]);
  int kmeansk = atoi(argv[2]);

  int r, c;
  float* data = FromFile(fileName, r, c);

  float* cpuLaplacian = (float*)malloc(c * c * sizeof(float));
  float* laplacian;
  float* gpuData;
  auto totalElapsed = 0.0f;

  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  checkCudaErrors(cudaMalloc((void**)&gpuData, r * c * sizeof(float)));
  checkCudaErrors(cudaMemcpy(gpuData, data, r * c * sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void**)&laplacian, c * c * sizeof(float)));
  checkCudaErrors(cudaMemset(laplacian, 0, c * c * sizeof(float)));
  CudaComputeLaplacian(laplacian, gpuData, c, r, knnk);
  checkCudaErrors(cudaMemcpy(cpuLaplacian, laplacian, c * c * sizeof(float), cudaMemcpyDeviceToHost));
  sdkStopTimer(&hTimer);
  auto dAvgSecs = (float)(1.0e-3 * (double)sdkGetTimerValue(&hTimer));
  printf("%.5f,", dAvgSecs);
  totalElapsed += dAvgSecs;

  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  void* eigenvalues = malloc(c * sizeof(float));
  void* eigenvectors = malloc(c * c * sizeof(float));
  void* eigenvectorsTmp = malloc(c * kmeansk * sizeof(float));

  memset(eigenvalues, 0, c * sizeof(float));
  memset(eigenvectors, 0, c * c * sizeof(float));

  ComputeEigens((void*)cpuLaplacian, c, &eigenvalues, &eigenvectors);
  SelectEigenVector((float*)eigenvectors, c, c, (float*)eigenvectorsTmp, kmeansk);
  sdkStopTimer(&hTimer);
  dAvgSecs = (float)(1.0e-3 * (double)sdkGetTimerValue(&hTimer));
  printf("%.5f,", dAvgSecs);
  totalElapsed += dAvgSecs;
  
  float* cpuEigenvectors;
  int* cpuCluster;

  checkCudaErrors(cudaMallocHost((void**)&cpuEigenvectors, c * kmeansk * sizeof(float)));
  checkCudaErrors(cudaMemcpy(cpuEigenvectors, eigenvectorsTmp, c * kmeansk * sizeof(float), cudaMemcpyHostToHost));

  checkCudaErrors(cudaMallocHost((void**)&cpuCluster, c * sizeof(int)));
  int* cpuGroup = cpuCluster;

  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  Cluster(cpuEigenvectors, c, kmeansk, cpuGroup, kmeansk);
  sdkStopTimer(&hTimer);
  dAvgSecs = (float)(1.0e-3 * (double)sdkGetTimerValue(&hTimer));
  totalElapsed += dAvgSecs;
  printf("%.5f,", dAvgSecs);
  printf("%.5f", totalElapsed);

  auto of = std::ofstream{ "../../scripts/LabeledPointsGPU.txt" };
  for (auto i = 0; i < c; ++i)
  {
    for (auto j = 0; j < r; ++j)
      of << data[i * r + j] << ",";
    of << cpuGroup[i] << "\n";
  }

  free(eigenvectorsTmp);
  free(eigenvectors);
  free(eigenvalues);
  cudaFree(laplacian);
  cudaFreeHost(cpuEigenvectors);
  cudaFreeHost(cpuGroup);
  free(data);
  free(cpuLaplacian);
  return 0;
}

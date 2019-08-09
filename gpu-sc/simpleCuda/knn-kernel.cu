/* Start Header
***************************************************************** /
/*!
\file knn-kernel.cu
\author Low Jin Liang Aaron
\brief
  Contains the implementation for k nearest neighbors on the gpu
*/
/* End Header
*******************************************************************/
#include "cublas.h"
#include "thrust\sort.h"
#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

#define KNN_BLOCK_SIZE       16
#define LAPLACIAN_BLOCK_SIZE 16

/*****************************************************************************/
/*
@function ComputeSquareDistanceTiled
  Computes the square distance between each point and places it in the dOut array
  Each block will compute its own block of distances. Tiled optimization

@param dOut
  Destination array of sq distances

@param dIn
  Input Samples

@param n
  Number of Samples

@param d
  Dimension of each sample
*/
/*****************************************************************************/
__global__ void ComputeSquareDistance(float* dOut, float* dIn, int n, int d)
{
  // Load values that will be reused
  __shared__ float blockA[KNN_BLOCK_SIZE][KNN_BLOCK_SIZE];
  __shared__ float blockB[KNN_BLOCK_SIZE][KNN_BLOCK_SIZE];
  
  // A is responsible for points indexed between aStart and aEnd
  auto aStart = blockIdx.x * blockDim.x;
  // B is responsible for points indexed between bStart and bEnd
  auto bStart = blockIdx.y * blockDim.y;

  auto ax = aStart + threadIdx.x;
  auto bx = bStart + threadIdx.y;

  auto sqDist = 0.0f;

  auto numBlocksVertical = (d - 1) / KNN_BLOCK_SIZE + 1;
  
  // Number of blocks that can be stored along the vertical dimension = gridDim.y
  // Therefore this loop runs for each block along the vertical dimension
  for(auto i = 0; i < numBlocksVertical; ++i)
  {
    // The i'th block on the vertical
    auto startY = i * KNN_BLOCK_SIZE;
    auto currY  = startY + threadIdx.y;

    // The first part of the algorithm has each thread responsible
    // for loading the values into blockA and blockB
    if(startY + threadIdx.y < d)
    {
      if(ax < n)
        blockA[threadIdx.y][threadIdx.x] = dIn[ax * d + currY];
      if(bx < n)
        blockB[threadIdx.y][threadIdx.x] = dIn[(bStart + threadIdx.x) * d + currY];
    }

    __syncthreads();

    // Since <a,a> = a1 * a1 + a2 * a2 + a3 * a3 + ... + ad * ad
    // We can compute the partial sum a1 * a1 + a2 * a2 + a3 * a3 + ... + ak * ak s.t k < d
    // Each thread is now responsible for computing the partial sum of their respective element
    // If the respective element is out of bounds, this loop can be skipped
    if(ax < n && bx < n)
      for(auto j = 0; j < KNN_BLOCK_SIZE; ++j)
      {
        auto diff = blockA[j][threadIdx.x] - blockB[j][threadIdx.y];
        sqDist += diff * diff;
      }
  }

  if(ax < n && bx < n)
  {
    dOut[ax * n + bx] = ax == bx ? INFINITY : sqDist;
  }
}

/*****************************************************************************/
/*
@SortDistances
Perform sorting of each column of the matrix according to distances. We have to
make a modification such that the indices are remembered as well
Each thread is responsible for sorting a single column.
Since k is expected to be small -> Insertion sort will be more efficient than quicksort

@param dist
The distance matrix

@param id
Output index array

@param n
Number of samples

@param k
'k' nearest neighbors to consider
*/
/*****************************************************************************/
__global__ void InsertionSortDistances(float* dist, int* idMat, int n, int k)
{
  // Get the index of the column that the current thread is responsible for
  auto col = blockIdx.x * blockDim.x + threadIdx.x;

  // IF col is out of bounds, then do nothing
  if (col < n)
  {
    auto id = &idMat[col * n];

    id[0] = 0;

    auto distCol = &dist[col * n];
    
    // Otherwise, sort column 'col'
    auto i = 1;
    while(i < n)
    {
    auto x = distCol[i];
    auto currIndex = i;
    auto j = i - 1;
    while(j >= 0 && distCol[j] > x)
    {
    distCol[j + 1] = distCol[j];
    id[j + 1] = id[j];
    --j;
    }
    distCol[j + 1] = x;
    id[j + 1] = currIndex;
    ++i;
    }
  }
}

/*****************************************************************************/
/*
@SortDistances
  Perform sorting of each column of the matrix according to distances. We have to 
  make a modification such that the indices are remembered as well
  Each thread is responsible for sorting a single column.
  Since k is expected to be small -> Insertion sort will be more efficient than quicksort

@param dist
  The distance matrix

@param id
  Output index array

@param n
  Number of samples

@param k
  'k' nearest neighbors to consider
*/
/*****************************************************************************/
__global__ void SortDistances(float* dist, int* idMat, int n, int k)
{
  // Get the index of the column that the current thread is responsible for
  auto col = blockIdx.x * blockDim.x + threadIdx.x;

  // IF col is out of bounds, then do nothing
  if(col < n)
  {
    auto id = &idMat[col * n];
    for(auto i = 0; i < n; ++i)
      id[i] = i;

    auto distCol = &dist[col * n];
    // Only care about the first k elements being sorted
    for (auto i = 0; i < k; ++i)
    {
      auto minIndex = i;
      for (auto j = i + 1; j < n; ++j)
      {
        if(distCol[j] < distCol[minIndex])
          minIndex = j;
      }
      auto tmp = distCol[minIndex]; 
      distCol[minIndex] = distCol[i];
      distCol[i] = tmp;

      auto tmpId = id[minIndex];
      id[minIndex] = id[i];
      id[i] = tmpId;
    }
  }
}

/*****************************************************************************/
/*
@function ComputeAdjacencyMatrix
  Computes the optimistic Adjacency Matrix in dOut. Each thread is responsible
  for a single column in the adjacency matrix

@param dOut
  The output adjacency matrix

@param nn
  The nearest neighbors matrix (k x n)

@param n
  The number of samples

@param k
  The number of neighbors
*/
/*****************************************************************************/
__global__ void ComputeAdjacencyMatrix(float* dOut, int* nn, int n, int k)
{
  // Get the column that the current thread is responsible for
  auto col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // If id is within bounds
  if(col < n)
  {
    auto nnCol = &nn[col * n];
    for(auto i = 0; i < k; ++i)
    {
      dOut[col * n + nnCol[i]] = dOut[col + n * nnCol[i]] = 1.0f;
    }
  }
}

/*****************************************************************************/
/*
@function ComputeLaplacianInPlace

@param d
  The laplacian matrix (input as adjacency matrix)

@param n
  The number of samples
*/
/*****************************************************************************/
__global__ void ComputeLaplacianInPlace(float* d, int n)
{
  // Column to sum
  auto x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x < n)
  {
    auto dCol = &d[x * n];

    for(auto i = 0; i < n; ++i)
    {
      if(i != x)
      {
        dCol[x] += dCol[i];
        dCol[i] = -dCol[i];
      }
    }
  }
}

/*****************************************************************************/
/*
@function CudaComputeLaplacian

@param dOut
  The computed laplacian matrix

@param dIn
  The input data set (d * n) column major

@param n
  Number of Samples

@param d
  Number of features

@param k
  Number of nearest neighbors to consider for KNN search
*/
/*****************************************************************************/
extern "C" void CudaComputeLaplacian(float* dOut, float* dIn, int n, int d, int k)
{
  auto knngrid            = dim3{ (unsigned)((n - 1) / KNN_BLOCK_SIZE + 1), (unsigned)((n - 1) / KNN_BLOCK_SIZE + 1), 1 };
  auto knnBlockDim        = dim3{ KNN_BLOCK_SIZE, KNN_BLOCK_SIZE, 1};
  auto singleAxisGridDim  = dim3{(unsigned)((n - 1) / KNN_BLOCK_SIZE + 1), 1, 1};
  auto singleAxisBlockDim = dim3{ KNN_BLOCK_SIZE, 1, 1};

  // Necessary for computing of k-nearest-neighbors
  int* indices;
  float* distances;
  checkCudaErrors(cudaMalloc((void**)&indices, n * n * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&distances, n * n * sizeof(int)));

  // Compute Square Distances
  ComputeSquareDistance<<<knngrid, knnBlockDim>>>(distances, dIn, n, d);
  cudaDeviceSynchronize();
  SortDistances<<<singleAxisGridDim, singleAxisBlockDim>>>(distances, indices, n, k);
  cudaDeviceSynchronize();
  ComputeAdjacencyMatrix<<<singleAxisGridDim, singleAxisBlockDim>>>(dOut, indices, n, k);
  cudaDeviceSynchronize();
  ComputeLaplacianInPlace<<<singleAxisGridDim, singleAxisBlockDim>>>(dOut, n);
  cudaDeviceSynchronize();

  cudaFree(distances);
  cudaFree(indices);
}

/*****************************************************************************/
/*
@function CudaTestAdjacency

@param dOut
  The computed adjacency matrix

@param dIn
  The input data set (d * n) column major

@param n
  Number of Samples

@param d
  Number of features

@param k
  Number of nearest neighbors to consider for KNN search
*/
/*****************************************************************************/
extern "C" void CudaTestAdjacency(float* dOut, float* dIn, int n, int d, int k)
{
  auto knngrid            = dim3{ (unsigned)((n - 1) / KNN_BLOCK_SIZE + 1), (unsigned)((n - 1) / KNN_BLOCK_SIZE + 1), 1 };
  auto knnBlockDim        = dim3{ KNN_BLOCK_SIZE, KNN_BLOCK_SIZE, 1};
  auto singleAxisGridDim  = dim3{(unsigned)((n - 1) / KNN_BLOCK_SIZE + 1), 1, 1};
  auto singleAxisBlockDim = dim3{ KNN_BLOCK_SIZE, 1, 1};

  // Necessary for computing of k-nearest-neighbors
  int* indices;
  float* distances;
  cudaMalloc((void**)&indices, n * n * sizeof(int));
  cudaMalloc((void**)&distances, n * n * sizeof(int));
  
  // Compute Square Distances
  ComputeSquareDistance<<<knngrid, knnBlockDim>>>(distances, dIn, n, d);
  cudaDeviceSynchronize();
  SortDistances<<<singleAxisGridDim, singleAxisBlockDim>>>(distances, indices, n, k);
  cudaDeviceSynchronize();
  ComputeAdjacencyMatrix<<<singleAxisGridDim, singleAxisBlockDim>>>(dOut, indices, n, k);
  cudaDeviceSynchronize();

  cudaFree(distances);
  cudaFree(indices);
}
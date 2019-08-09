/* Start Header
***************************************************************** /
/*!
\file knn-kernel.cu
\author Koh Wen Lin
\brief
  Contains the implementation for kmeans clustering on the gpu.
*/
/* End Header
*******************************************************************/
#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#define KMEAN_BLOCK_SIZE 32
#define KMEAN_BLOCK_SIZE_1D KMEAN_BLOCK_SIZE * KMEAN_BLOCK_SIZE


__device__ float distanceSq(const float* a, const float* b, size_t d)
{
  float result = 0.0f;
  for(int i = 0; i < d; ++i)
    result += (a[i] - b[i]) * (a[i] - b[i]);

  return result;
}

__device__ float distance(const float* a, const float* b, size_t d)
{
  return sqrt(distanceSq(a, b, d));
}

__global__ void Group(float* dIn, unsigned n, unsigned d, float* dMeanIn, unsigned k, int* dGroupOut)
{
  // Each thread performs 1 grouping of data in a cluster
  // Each tid represents 1 data in dataset, 1 data consist of n feature
  extern __shared__ float sMeanData[];  // Dynamic allocated shared memory enough to store block-size amount of data and centroids.

  float* sMean = sMeanData;    // ptr to shared memory storing centroids
  float* sData = sMean + k * d;        // ptr to shared memory storing block-level data samples

  int tx = threadIdx.x;

  // Compute index of data for this thread
  int tid = blockIdx.x * blockDim.x + tx;

  if (tid >= n) 
    return;

  // Use first k-th thread to load centroids into shared memory
  if(tx < k)
    memcpy(&sMean[tx * d], &dMeanIn[tx * d], d * sizeof(float));

  // Use each thread in block level to perform 1 global load of its own features
  if(tid < n)
    memcpy(&sData[tx * d], &dIn[tid * d], d * sizeof(float));

  // Ensure data loaded by all threads
  __syncthreads();

  // Perform grouping for each data per thread
  int nearest = 0;
  float shortestDist = INFINITY;

  for(int i = 0; i < k; ++i)
  {
    float distSq = distanceSq(&sData[tx * d], &sMean[i * d], d);

    if(distSq < shortestDist)
    {
      nearest = i;
      shortestDist = distSq;
    }
  }

  dGroupOut[tid] = nearest;
}

__global__ void Mean(float* dIn, unsigned n, unsigned d, int* dGroupIn, float* dMeanIn, unsigned k, int* count)
{
  // Each thread block to perform its own summation internally(Reduction), then, each thread block will add its result into global counter and sum
  extern __shared__ float sDataSumGroupCount[]; // Dynamic allocated shared memory enough to store block-size amount of data and sum of cluster, group and count.

  float* sData = sDataSumGroupCount;
  float* sSum = sData + KMEAN_BLOCK_SIZE_1D * d;
  int* sGroup = (int*)&sDataSumGroupCount[(k + KMEAN_BLOCK_SIZE_1D) * d];
  int* sCount = sGroup + KMEAN_BLOCK_SIZE_1D;

  const int tx = threadIdx.x;
  int tid = blockIdx.x * blockDim.x + tx;

  if(tid >= n)
    return;

  // Clear shared memory
  if(tx < k)
  {
    for(int i = 0; i < d; ++i)
      sSum[tx * d + i] = dMeanIn[tx * d + i];
    sCount[tx] = count[tx] = 0.0f;
  }

  // Each thread perform 1 global load for all its feature and its group index
  memcpy(&sData[tx * d], &dIn[tid * d], d * sizeof(float));
  sGroup[tx] = dGroupIn[tid];

  // Clear old mean
  memset(dMeanIn, 0, k * d * sizeof(float));

  // Ensure all data relavant to block is loaded
  __syncthreads();

  int clusterId = sGroup[tx];

  for(int i = 0; i < d; ++i)
    atomicAdd(&sSum[clusterId * d + i], sData[tx * d + i]);
  atomicAdd(&sCount[clusterId], 1);

  __syncthreads();

  if(tx == 0)
  {
    for(int i = 0; i < k * d; ++i)
      atomicAdd(&dMeanIn[i], sSum[i]);

    for(int i = 0; i < k; ++i)
      atomicAdd(&count[i], sCount[i]);
  }
}

__global__ void MeanUpdate(float* dMeanIn, unsigned k, unsigned d, int* count)
{
  float ooc = 1.0f / max(1, count[threadIdx.x]);
  for(int i = 0; i < d; ++i)
    dMeanIn[threadIdx.x * d + i] *= ooc;
}

extern "C" void ClusterGPU(
  float *dIn,
  unsigned n,
  unsigned d,
  int *dGroupOut,
  float *dMean,
  unsigned k,
  int* dCountOut
)
{
  dim3 dimBlock(KMEAN_BLOCK_SIZE_1D);
  dim3 dimGrid((unsigned)ceil(n / (float)KMEAN_BLOCK_SIZE_1D));

  // K means clustering iteration
  int NumOfIteration = 2000;
  while(--NumOfIteration)
  {
    // Group data in GPU
    Group<<<dimGrid, dimBlock, (k + KMEAN_BLOCK_SIZE_1D) * d * sizeof(float)>>>(
      dIn,
      n, d,
      dMean,
      k,
      dGroupOut
      );

    cudaDeviceSynchronize();

    Mean<<<dimGrid, dimBlock, (k + KMEAN_BLOCK_SIZE_1D) * d * sizeof(float) + (k + KMEAN_BLOCK_SIZE_1D) * sizeof(int)>>>(
      dIn,
      n, d,
      dGroupOut,
      dMean,
      k,
      dCountOut
      );
    
    cudaDeviceSynchronize();

    MeanUpdate<<<1, k>>>(
      dMean,
      k, d,
      dCountOut
      );

    cudaDeviceSynchronize();
  }
}


extern "C" void Cluster(float* h_DataIn, unsigned n, unsigned d, int* h_GroupOut, int k)
{
  float* d_Cluster;

  float* d_DataIn; // Device data input
  unsigned inCount = n * d;
  size_t inByteCount = inCount * sizeof(float);

  int* d_GroupOut; // Device group data storage
  size_t groupOutByteCount = n * sizeof(int);

  float* d_Mean; // Device mean data
  size_t meanByteCount = k * d * sizeof(float);

  int* d_CountOut; // Device cluster group count data
  size_t countByteCount = k * sizeof(int);

  // Allocate device memory
  checkCudaErrors(cudaMalloc((void**)&d_Cluster, inByteCount + groupOutByteCount + meanByteCount + countByteCount));

  d_DataIn = d_Cluster;
  d_Mean = d_DataIn + inCount;
  d_GroupOut = (int*)(d_Mean + k * d);
  d_CountOut = d_GroupOut + n;

  // Copy host to device
  checkCudaErrors(cudaMemcpy(d_DataIn, h_DataIn, inByteCount, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Mean, h_DataIn, meanByteCount, cudaMemcpyHostToDevice));

  // Perform kernel k-means cluster
  ClusterGPU(d_DataIn, n, d, d_GroupOut, d_Mean, k, d_CountOut);

  // Copy device to host
  checkCudaErrors(cudaMemcpy(h_GroupOut, d_GroupOut, groupOutByteCount, cudaMemcpyDeviceToHost));

  // Release resources
  checkCudaErrors(cudaFree(d_Cluster));
}
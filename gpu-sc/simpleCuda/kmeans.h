#pragma once

extern "C" void ClusterGPU(
  float *dIn,
  unsigned n,
  unsigned d,
  int *dGroupOut,
  float *dMean,
  unsigned k,
  int* dCountOut
);

extern "C" void Cluster(float* h_DataIn, unsigned n, unsigned d, int* h_GroupOut, int k);
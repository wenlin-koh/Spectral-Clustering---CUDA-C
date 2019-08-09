#pragma once

extern "C" void CudaTestAdjacency(float* dOut, float* dIn, int n, int d, int k);
extern "C" void CudaComputeLaplacian(float* dOut, float* dIn, int n, int d, int k);
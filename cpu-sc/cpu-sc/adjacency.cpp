#include "adjacency.h"
#include <mkl.h>
#include <malloc.h>
#include <algorithm>
#include <cassert>

/*****************************************************/
/*
	@function DistanceMatrix
		Compute the Similarity graph based on k-nearest neighbors

	@param dOut
		Pointer to the output similarity graph (n x n)

	@param dIn
		Pointer to the input data set (d x n)

	@param n
		The number of samples

	@param d
		The number of features

	@param k
		The number of nearest neighbors
*/
/*****************************************************/
void Adjacency(float* dOut, float* dIn, int n, int d, int k)
{
	// @TODO: Fix case when k is greater than n
	// A possible shortcut to guarantee symmetry is to just assume undirected that is to denote (i, j) and (j, i)
	// as edges so long as either j or i is part of the set of knn of the other

	void* buf = _malloca(n * (sizeof(int) + sizeof(float)));
	assert(buf);

	int* nn = (int*)buf;
	float* distances = (float*)&(((int*)buf)[n]);
  float* tmp = (float*)_malloca(d * sizeof(float));
	
	// For each vertex v_i
	for (auto i = 0; i < n; ++i)
	{
		int c = 0;
		std::fill(distances, distances + n, FLT_MAX);

		for (auto j = 0; j < n; ++j)
		{
			// Ignore self 
			if (i != j)
			{
				// We compute distance through euclidean norm
        vsSub(d, &dIn[i * d], &dIn[j * d], tmp);
				auto dist = cblas_sdot(d, tmp, 1, tmp, 1);
				if (c < k)
				{
					// Store distance
					distances[c] = dist;
					// Store index of nearest neighbor
					nn[c] = j;
					++c;
				}
				else
				{
					auto maxIt = std::max_element(distances, distances + k);
          // Replace distance
					if (dist < *maxIt)
					{
						auto id = maxIt - distances;
						// Replace index of nearest neighbor
						nn[id] = j;
						// Replace distance
						distances[id] = dist;
					}
				}
			}
		}

		// Currently set to 1 to show an edge between the k nearest neighbors
		// Set both to force an undirected graph ( Symmetric matrix )
		for (auto j = 0; j < k; ++j)
			dOut[nn[j] * n + i] = dOut[i * n + nn[j]] = 1.f;
	}

	_freea(nn);
  _freea(tmp);
}

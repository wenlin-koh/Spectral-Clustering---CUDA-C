#include "laplacian.h"

/*****************************************************/
/*
	@function Laplacian
		Compute Laplacian matrix with given adjacency matrix

	@param l
		Output Laplacian (n x n)

	@param adj
		Input Adjacency Matrix (n x n)

	@param n
		Dimension of Matrix
*/
/*****************************************************/
void Laplacian(float* l, float* adj, int n)
{
	for (auto i = 0; i < n; ++i)
	{
		for (auto j = 0; j < n; ++j)
		{
			// Sum degree
			l[i * n + i] += adj[i * n + j];
			// Off-diagonal terms
			if(i != j)
				l[i * n + j] = -adj[i * n + j];
		}
	}
}

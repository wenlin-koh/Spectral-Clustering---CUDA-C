#include "load.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>

/*****************************************************/
/*
	@function FromFile

	Load the data stored in the path variable as the designer
	matrix

	FILE FORMAT:
	Should begin with the number of features (d) and the number of samples (n)
	separated by a single space
	The subsequent n lines of the file should each contain d values that signify
	the value of each feature at that respective index

	d n
	n11 n12 n13 n14 ... n1d
	n21 n22 n23 n24 ... n2d
	...

	@param path
	Path to file

	@param r
	Output variable to store the number of rows

	@param c
	Output variable to store the number of columns

	@returns
	Pointer to allocated array storing the designer matrix
	It is the users responsibility to free the memory after use
	Returns null if failed
*/
/*****************************************************/
float* FromFile(const char* path, int& r, int& c)
{
	// Open file
	auto f = std::ifstream{path};

	if (f.is_open())
	{
		size_t d, n;
		f >> d >> n;
		auto m = (float*)malloc(d * n * sizeof(float));
		if (!m)
		{
			std::cerr << "Failed to allocate memory for designer matrix!\n";
			return nullptr;
		}
		for (auto i = 0; i < n; ++i)
			for (auto j = 0; j < d; ++j)
				f >> m[i * d + j];
		r = (int)d;
		c = (int)n;
		return m;
	}

	std::cerr << "Failed to open " << path << "\n";
	return nullptr;
}


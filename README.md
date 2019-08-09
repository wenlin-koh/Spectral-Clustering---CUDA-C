# Spectral-Clustering---CUDA-C

# Instructions
Program requires Intel MKL library to be installed for Windows x86_64
Ensure that visual studio property pages is set to use Intel MKL sequentially
Intel MKL library download: https://software.intel.com/en-us/mkl/choose-download
GPU Version expects CUDA Toolkit 9.2 to be installed

# Usage
CPU Version: ./Prog.exe <k-nearest neighbors> <k-clusters> <data filename> (VS2017)
GPU Version: ./Prog.exe <k-nearest neighbors> <k-clusters> <data filename> (VS2015)

# Plotting Results
The results will be placed in "Scrips/" plot.py in the same folder can be used to 
generate a visualization of the results. plot.py requires matplotlib.
Usage: .\plot.py <filename>

# Test cases
We have generated 6 datasets for testing - "Moons_500", "Circles_500", "Circles_5000", "Circles_10000".
The expected k-nearest neighbors for Moons_500 is 5 while the expected k-nearest neighbors for circles
is 8. All datasets expect 2 k-clusters. (For optimal results)
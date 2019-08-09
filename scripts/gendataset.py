#!/bin/env python
"""
gendataset.py : Python Script to generate data sets for spectral clustering project

Loads data from sklearn.datasets then converts the data and outputs a text file in our
custom format
"""
from sklearn.datasets import make_circles, make_blobs, make_moons
import matplotlib.pyplot as plt
import sys

datasets = {
    "Circles" : lambda x : make_circles(x, noise = 0.08, factor = 0.3),
    "Moons" : lambda x : make_moons(x, noise = 0.07),
    "Blobs" : make_blobs
}

def main():
    argc = len(sys.argv)
    if argc < 3 :
        raise RuntimeError("Usage: ./gendataset.py <Circles/Moons/Blobs> <integer arguments>")
    # Load Data Sets
    X, y = datasets[sys.argv[1]](*[int(x) for x in sys.argv[2:]])
    with open(sys.argv[1] + ".txt", "w+") as f:
        f.write(str(X.shape[1]) + " " + str(X.shape[0]) + "\n")
        # For each sample
        for i in range(X.shape[0]):
            line = str()
            for j in range(X.shape[1]):
                line = line + str(X[i,j]) + " "
            f.write(line + "\n")
    with open(sys.argv[1] + "-Labels.txt", "w+") as f:
        f.write("1 " + str(len(y)) + "\n")
        # For each sample
        for i in range(len(y)):
            f.write(str(y[i]) + "\n")
    plt.scatter([x[0] for x in X], [x[1] for x in X])
    plt.show()


if __name__ == "__main__":
    main()

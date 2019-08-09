#!/usr/bin/bash

EXPNAME = BASH_ARGV[1]
CPUEXEC = BASH_ARGV[2]
GPUEXEC = BASH_ARGV[3]

echo "Remove Previous Logs"
rm -f Logs_${EXPNAME}.txt

# Filename array
FILENAMES = (
  "../../data/Moons_500.txt",
  "../../data/Moons_2000.txt",
  "../../data/Circles_500.txt",
  "../../data/Circles_5000.txt",
  "../../data/Circles_10000.txt"
)

# K nearest neighbor k value
KNN = (
  5,
  5,
  8,
  8,
  8
)

# K means clustering k value
KMC = (
  2,
  2,
  2,
  2,
  2
)

echo "Begin Benchmark Test Sequence for Spectral Clustering!"
NUMTESTS = 1 # ${#FILENAMES[@]}
echo -n "Number of Tests: "
echo ${NUMTESTS}

echo "------------------------------------------------------"
echo "CPU Spectral Clustering Benchmarks"

for((i=0; i < NUMTESTS; ++i)); do
    ./${CPUEXEC} KNN[i] KMC[i] FILENAMES[i] | tee -a Logs_${EXPNAME}.txt

echo "GPU Spectral Clustering Benchmarks"

for((i=0; i < NUMTESTS; ++i)); do
    ./${GPUEXEC} KNN[i] KMC[i] FILENAMES[i] | tee -a Logs_${EXPNAME}.txt

echo "Done"
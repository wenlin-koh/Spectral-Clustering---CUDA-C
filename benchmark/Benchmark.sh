#!/usr/bin/bash
EXPNAME=${BASH_ARGV[2]}
CPUEXEC=${BASH_ARGV[1]}
GPUEXEC=${BASH_ARGV[0]}
echo "CPUEXEC: "
echo ${CPUEXEC}
echo "GPUEXEC: "
echo ${GPUEXEC}
echo "Remove Previous Logs"
rm -f Logs_${EXPNAME}.txt
# Filename array
FILENAMES=(
  "Moons_500.txt"
  "Moons_2000.txt"
  "Circles_500.txt"
  "Circles_5000.txt"
  "Circles_10000.txt"
)
# K nearest neighbor k value
KNN=(
  5
  5
  8
  8
  8
)
# K means clustering k value
KMC=(
  2
  2
  2
  2
  2
)
echo "Begin Benchmark Test Sequence for Spectral Clustering!"
NUMTESTS=1 # ${#FILENAMES[@]}
echo "Number of Tests: "
echo ${NUMTESTS}
echo "------------------------------------------------------"
echo "CPU Spectral Clustering Benchmarks"
for((i=0;i<${NUMTESTS};++i));do
    ./${CPUEXEC} ${KNN[$i]} ${KMC[$i]} ${FILENAMES[$i]} | tee -a Logs_${EXPNAME}.txt
  done
echo "GPU Spectral Clustering Benchmarks"
for((i=0;i<${NUMTESTS};++i));do
    ./${GPUEXEC} ${KNN[$i]} ${KMC[$i]} ${FILENAMES[$i]} | tee -a Logs_${EXPNAME}.txt
  done
echo "Done"
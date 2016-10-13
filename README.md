# k||means

A library to compure k-means in the following settings:
1. Non-Uniform Memory Access (NUMA) machines. A NUMA-optimized multithreaded
    ,via [p-threads](https://computing.llnl.gov/tutorials/pthreads/) implemation for shared-memory linux
    systems
2. An MPI-based implementation of the k||means algorithm.

These implementaions are incarnations of algorithms
[our publication](https://arxiv.org/abs/1606.08905). These implementations have the ability read from local disk, Amazon S3 and HDFS.

## System Requirements
TODO

## Installation
The following is Tested on Ubuntu 14.04:


### Auto-Install
`./boostrap.sh`

## Manual installation -- Dependencies
TODO

## Data format conversion
TODO

### Usage
* NUMA-k||means:
    `./kpmeans datafile nsamples dimension k -t random -i 10 -p -m`

* OMP-k||means:
    `./kpmeans datafile nsamples dimension k -t random -i 10 -m`

* MPI-k||means:
    `mpirun.mpich -n nproc ./dist_kmeans -f datafile_cw.dat -k k -n nsamples -d dimension -I random -i 10`

* SEM-k||means:
    `TODO`

# k||means

A library to compure k-means in the following settings:
1. Non-Uniform Memory Access (NUMA) machines. A NUMA-optimized multithreaded
    ,via [p-threads](https://computing.llnl.gov/tutorials/pthreads/) implemation for shared-memory linux
    systems
2. An MPI-based implementation of the k||means algorithm.

These implementaions are incarnations of algorithms
[our publication](https://arxiv.org/abs/1606.08905). These implementations have the ability read from local disk, Amazon S3 and HDFS.

## Installation

TODO

## Data format conversion

TODO

### Usage
* NUMA-k||means  
	`./exec TODO`

* MPI-k||means  
`mpirun -n #procs `
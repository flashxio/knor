# k||means

A library to compure k-means in the following settings:

1. Non-Uniform Memory Access (NUMA) machines. A NUMA-optimized multithreaded
    ,via [p-threads](https://computing.llnl.gov/tutorials/pthreads/)
    implemation for shared-memory linux systems
2. An MPI-based implementation of the ||Lloyds algorithm.
3. Options to use a scalable adaption of
[Elkan's](http://users.cecs.anu.edu.au/~daa/courses/GSAC6017/kmeansicml03.pdf)
agorithm that *can* drastically reduce the number of distance computations
required.

These implementations are incarnations of algorithms in
[our publication](https://arxiv.org/abs/1606.08905). These implementations
have the ability read from local disk, Amazon S3 and HDFS.

## System Requirements
- Linux
- At least **4 (GB) of RAM**
- Administrative privileges on the machine

## Installation
The following is Tested on **Ubuntu 14.04 - 16.04**. We require a fairly
modern compiler to take advantage of compile time optimizations:

### Auto-Install
`./boostrap.sh`

### Python bindings

#### Requirements:
- Cython 0.21+

### Usage
Assume the following:
	- `k` is the number of clusters
	- `dim` is the dimensionality of the features
	- `nsamples` is the number of samples

* NUMA-k||means:
    `./kpmeans datafile nsamples dim k -t random -i 10 -p -m`

* OMP-k||means:
    `./kpmeans datafile nsamples dim k -t random -i 10 -m`

* MPI-k||means:
    `mpirun.mpich -n nproc ./dist_kmeans -f datafile_cw.dat\`
   	`-k k -n nsamples -d dim -I random -i 10`

* SEM-k||means:
    `TODO`

## Data format conversion
We provide some lightweight fast utilities to convert data from
common formats to our own. Below we document their use:
TODO

## Publications

Mhembere, D., Zheng, D., Vogelstein, J. T., Priebe, C. E., & Burns, R. (2016).
NUMA-optimized In-memory and Semi-external-memory Parameterized Clustering.
arXiv preprint arXiv:1606.08905.


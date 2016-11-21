# knor

The K-means NUMA Optimized Routine library or **knor** is a
library for computing k-means in parallel with accelerations for
Non-Uniform Memory Access (NUMA) architectures in the following settings.

1. On a single machine with all data In-memory, *knor-IM*.
2. In a cluster in distributed memory *knor-DM*.
3. On a single machine with some data in-memory and the rest on SSDs i.e.,
Semi-External Memory, *knor-SEM*.

knor can use a scalable adaption of
[Elkan's](http://users.cecs.anu.edu.au/~daa/courses/GSAC6017/kmeansicml03.pdf)
algorithm that *can* drastically reduce the number of distance computations
required for k-means.

## knor backbone

**knor** relies on the following:
- [p-threads](https://computing.llnl.gov/tutorials/pthreads/) for
multithreading.
- [numa](https://linux.die.net/man/3/numa) and
[hwloc](https://linux.die.net/man/7/hwloc) for NUMA optimization.
- [FlashGraph](https://github.com/flashxio/FlashX) for a semi-external memory
vertex-centric interface.

These modules are incarnations of algorithms in
[our publication](https://arxiv.org/abs/1606.08905). These implementations
have the ability read from local disk, Amazon S3 and HDFS.

## System Requirements
- Linux
- At least **4 (GB) of RAM**
- Administrative privileges

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
    - `$INSTALL_HOME` is the directory where you clone knor.

To run modules from the `$INSTALL_HOME/exec` directory, you may do the
following:

* knor-IM:
    `./kpmeans datafile nsamples dim k -t random -i 10 -p -m`

*For comparison run our algorithms using [OpenMP](http://www.openmp.org/)*
    `./kpmeans datafile nsamples dim k -t random -i 10 -m`

* knor-DM:
    `mpirun.mpich -n nproc ./dist_kmeans -f datafile_cw.dat\`
   	`-k k -n nsamples -d dim -I random -i 10`

* knor-SEM:
    `TODO`

## Data format conversion
We provide some lightweight fast utilities to convert data from
common formats to our own. Below we document their use:
TODO

## Publications

Mhembere, D., Zheng, D., Vogelstein, J. T., Priebe, C. E., & Burns, R. (2016).
NUMA-optimized In-memory and Semi-external-memory Parameterized Clustering.
arXiv preprint arXiv:1606.08905.


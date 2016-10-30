# k||means (Alpha Release)

A Parallel and Distributed library to compute **k-means**. This library outperforms commercial products like
Turi (Graphlab, Dato) and MLlib often by an order of magnitude or more. The library contains:

1. Non-Uniform Memory Access (NUMA) machine optimizations. A NUMA-optimized multithreaded
    ,via [p-threads](https://computing.llnl.gov/tutorials/pthreads/)
    implemation for shared-memory linux systems.
2. An MPI-based distributed-parallel implementation of the ||Lloyds algorithm
	where each *thread* lauched by an MPI *process* then executes our NUMA optimizations.
3. Options to use a scalable lightweight adaption of
[Elkan's](http://users.cecs.anu.edu.au/~daa/courses/GSAC6017/kmeansicml03.pdf) 
agorithm that *can* drastically reduce the number of distance computations
required **without the memory blowup** that traditionally ailes Elkan's alogrithm.
4. **NOTE: Under migration** -- A semi-external memory implementation that scales well beyond the size of RAM and is geared
toward high-io enabled, stand-alone, thick nodes. Performance is comparable to in-memory
even when the data is significantly larger than RAM.

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

### Usage
Assume the following:
	- `k` is the number of clusters
	- `dim` is the dimensionality of the features
	- `nsamples` is the number of samples

* NUMA-k||means:
    ```bash
    ./kpmeans datafile nsamples dim k -t [random|forgy|kmeanspp] \
    -i 10 -p [-T,-i,-C,-l,-d,-m,-N]
    ```

* OMP-k||means:
    ```bash
    ./kpmeans datafile nsamples dim k -t [random|forgy|kmeanspp] \
    -i 10 [-T,-i,-C,-l,-d,-m,-N]
    ````

* MPI-k||means:
    ```bash
    mpirun.mpich -n nproc -host <workerip1>[,workerip2,...] \
    --map-by ppr:1:node \
    ./dist_kmeans datafile_cw.dat \
    k nsamples dim -t [random|forgy|kmeanspp] [-T,-i,-C,-l,-d,-m,-N]
    ```

* SEM-k||means:
    `TODO:` Migration from [FlashX branch](https://github.com/zheng-da/FlashX/tree/disa-graph-attr)
    in progress

## Data format conversion
We provide some utilities to convert data from common formats to our own which is simply **row-major flat binary**
with **no headers**. The `convert_matrix`utility obtained when [convert_matrix.cpp](https://github.com/disa-mhembere/k-par-means/blob/master/utils/convert_matrix.cpp) is compiled has several options. Execute `convert_matrix` to see options.

## Publications
Mhembere, D., Zheng, D., Vogelstein, J. T., Priebe, C. E., & Burns, R. (2016).
NUMA-optimized In-memory and Semi-external-memory Parameterized Clustering.
arXiv preprint arXiv:1606.08905.


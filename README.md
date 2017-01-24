[![Build
Status](https://travis-ci.org/disa-mhembere/knor.svg?branch=master)](https://travis-ci.org/disa-mhembere/knor)

![logo](https://docs.google.com/drawings/d/1wDfKYzNoYk4xmtgFbN2VrQJQemGQip_0BlOGkP9E87U/pub?w=480&amp;h=360)
# knor

The K-means NUMA Optimized Routine library or **knor** is a
library for computing k-means in parallel with accelerations for
Non-Uniform Memory Access (NUMA) architectures in the following settings.

*knor* can perform at 10-100X the speed of popular packages like Spark's MLlib, Dato (GraphLab) and H<sup>2</sup>O.

1. On a single machine with all data In-memory, *knori*.
2. In a cluster in distributed memory *knord*.
3. On a single machine with some data in-memory and the rest on SSDs i.e.,
Semi-External Memory, *knors*.

knor can use a **scalable** adaption of
[Elkan's](http://users.cecs.anu.edu.au/~daa/courses/GSAC6017/kmeansicml03.pdf)
algorithm that *can* drastically reduce the number of distance computations
required for k-means without the memory bloat traditionally associated with
Elkan's algorithm.

## knor backbone

**knor** relies on the following:

- [p-threads](https://computing.llnl.gov/tutorials/pthreads/) for
multithreading.
- [numa](https://linux.die.net/man/3/numa) and
[hwloc](https://linux.die.net/man/7/hwloc) for NUMA optimization.
- [FlashGraph](https://github.com/flashxio/FlashX) for a semi-external memory
vertex-centric interface.

These modules are incarnations of algorithms in
[our publication](https://arxiv.org/abs/1606.08905).

## System Requirements
- Linux
- At least **4 (GB) of RAM**
- Administrative privileges

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
- `$KNOR_HOME` is the directory created when *knor* was cloned from Github.
- `nproc` is the number of MPI processes to use for *knord*.
- `nthread` is the number of thread per process. For *knori* this is the total
number of threads of execution. **NOTE:** For *knord* the total number of threads of
executions is `nproc * nthread`.

To run modules from the `$KNOR_HOME` directory, you may do the following:

#### knori

For a help message and to see valid flags:

```
exec/knori
```

An example of how to process with file `datafile.bin`:

```
exec/knori datafile.bin nsamples dim k -t random -T 2 -i 10 -o outdir
```

For comparison (slower speed) run our algorithm using
[OpenMP](http://www.openmp.org/) as follows:

```
exec/knori datafile.bin nsamples dim k -t random -T nthread -i 10 -o outdirOMP -O
```

It is also possible to **disable** computataion pruning i.e., using *Minimal*
triangle inequality algorithm by using the `-P` flag.

#### knord

For a help message and to see valid flags:

```
exec/knord
```

An example of how to process with file `datafile.bin`:
```
mpirun.mpich -n nproc exec/knord datafile nsamples dim k \
    -t random -i 10 -T nthread -o outdir
```

See the [mpirun](https://www.open-mpi.org/doc/v2.0/man1/mpirun.1.php) list of
flags like to allow your processes to distribute
correctly across a cluster. Flags of note are:
- `--map-by ppr:n:node`, determining how many processes are mapped to a node.
- `--map-by ppr:n:socket`, determining how processes per socket.

#### knors:
```
TODO
```

## Data format

We use two different formats dependent upon if you operate fully in-memory or
use semi-external memory.

### *knori* and *knord*

In-memory and distributed memory routines (*knori*, *knord*) use row-major
binary matrices of type `double` i.e., 16 Bytes per entry. For example the
matrix:

```
1 2 3 4
5 6 7 8
```

Must be written out in row-major form i.e., the bytes would be organized such
that on disk they would look like:

```
1 2 3 4 5 6 7 8
```

**NOTE:** The row-major data on disk will be **non-human readable** since it is
in binary format.

### *knors*

Semi-external memory (*knors*) data is stored in row-major format with
a leading 4KB FlashGraph header. You can convert binary data to *knors*
format using:

- TODO: coming soon ...

### Data Conversion

We provide a single threaded data conversion script to convert data from one
format to the other.

#### Plain text (space separated values) to row-major binary

We consider a file named `example.txt` in our *knor* root directory,
`$KNOR_HOME` with the following content:

```
1 2 3 4
5 6 7 8
```

To convert this file from text readable (tsv) format one would do the following:

```
$ cd $KNOR_HOME
$ utils/convert_matrix # gives help on how to use the converter

usage: ./convert_matrix in_filename in_format [text/knori/knord/knors]\
    out_filename out_format[text/knori/knord/knors] nrow ncol

$ utils/convert_matrix example.txt text example.dat knori 2 4
```

### Points of note

- Different *knor* routines **may** not produce the same result
dependent upon which initialization routine is used, **but** every routine
will give the same answer over multiple runs.
- Found a bug? Please [create an issue](https://github.com/disa-mhembere/knor/issues/new) on Github.

## Publications

Mhembere, D., Zheng, D., Vogelstein, J. T., Priebe, C. E., & Burns, R. (2017).
knor: A NUMA-optimized In-memory, Distributed and Semi-external-memory k-means
Library. arXiv preprint arXiv:1606.08905.

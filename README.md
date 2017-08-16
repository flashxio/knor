[![Build
Status](https://travis-ci.org/flashxio/knor.svg?branch=master)](https://travis-ci.org/flashxio/knor)

![logo](https://docs.google.com/drawings/d/1JRW7oklR9yem5-w4Lz4mF2-AM8L0e_yH9Zc-HHJMFeU/pub?w=280&h=360)
# knor

The k-means NUMA Optimized Routine library or **knor** is a
highly optimized and fast library for computing k-means in
parallel with accelerations for Non-Uniform Memory Access (NUMA) architectures.

*knor* can perform at **10-100 times** the speed of popular packages like Spark's MLlib, Dato (GraphLab) and H<sup>2</sup>O.

*knor* can operate:

1. On a single machine with all data in-memory -- **knori**.
2. In a cluster in distributed memory -- **knord**.
3. On a single machine with some data in-memory and the rest on disk
(Semi-External Memory) -- **knors**.

These modules are incarnations of algorithms in
[our publication](https://arxiv.org/abs/1606.08905) which will appear in the
annuls of HPDC 2017.

By default *knor* can use a **scalable** adaption of
[Elkan's](http://users.cecs.anu.edu.au/~daa/courses/GSAC6017/kmeansicml03.pdf)
algorithm that *can* drastically reduce the number of distance computations
required for k-means without the memory bloat traditionally associated with
Elkan's algorithm.

# Python

We expose the **knori** to users through `Python`. We support Linux and Mac OSX.

For instructions on how to install the `Python` package please see the
[Python repository](https://github.com/flashxio/knorPy).

In most cases all that is necessary is:

```
pip install knor
```

# R

We expose the **knori** to users through `R`. We support Linux, Mac OSX, and **Windows**.

For instructions on how to install the `R` package please see the
[R repository](https://github.com/flashxio/knorR).

In most cases all that is necessary is:

```
install.packages("knor")
```

## knor's backbone

**knor** relies on the following:

- [p-threads](https://computing.llnl.gov/tutorials/pthreads/) for
multithreading.
- [numa](https://linux.die.net/man/3/numa) for NUMA optimization.
- [FlashGraph](https://github.com/flashxio/FlashX) for a semi-external memory
vertex-centric interface.

## System Requirements

- Linux or Mac OS 10.*
- At least **4 (GB) of RAM**
- Administrative privileges

## Installation

The following is Tested on **Ubuntu 14.04 - 16.04**. We require a fairly
modern compiler to take advantage of compile time optimizations:

### Native Auto-Install

We assume users have [`git`](https://git-scm.com/) already installed.

```
git clone --recursive https://github.com/flashxio/knor.git
cd knor
./bootstrap.sh
```

### Docker -- Run in a Container

Users can also choose to use a Docker container to run *knor*.

**Mac Docker Users NOTE:**

* You must use the `-O` flag for *knori* and *knord* to use
OpenMP due to a lack of hardware support for our optimizations.
* *knors* for Mac is unsupported due to the lack of support for low-level
interfaces.

#### Quick docker image (Stable release)

Users can get a version that is stable, but possibly not up to date as follows:

```
docker run -i -t flashxio/knor
```

If you already have the *knor* docker image then updating to the newest stable
release can be done as follows:

```
docker pull flashxio/knor
```

#### Manual build (Bleeding edge release)

Users may also obtain a bleeding-edge up to date stable docker image by
building the image themselves. We assume your
OS has [`curl`](https://linux.die.net/man/1/curl) and
[`docker`](https://docs.docker.com/engine/installation/) already installed.

```
curl -O https://raw.githubusercontent.com/flashxio/knor/master/Dockerfile
docker build -t knor .
docker run -i -t knor
```

### Usage (Running *knor*)

Assume the following:

- `k` is the number of clusters
- `dim` is the dimensionality of the features
- `nsamples` is the number of samples
- `$KNOR_HOME` is the directory created when *knor* was cloned from Github.
- `nproc` is the number of MPI processes to use for *knord*.
- `nthread` is the number of threads per process. For *knori* this is the total
number of threads of execution. **NOTE:** For *knord* the total number of threads of
executions is `nproc * nthread`.

To run modules from the `$KNOR_HOME` directory, you may do the following:

### Test Data

We maintain a very small dataset for testing:
[`$KNOR_HOME/test-data/matrix_r50_c5_rrw.bin`](https://github.com/flashxio/knor/blob/master/test-data/matrix_r50_c5_rrw.bin?raw=true).

This dataset has:

- `nsamples` = `50`
- `dim` = `5`

**NOTE: knor will fail if you provide more units of parallelism than the
`nsamples` in the dataset. So use small `-t` and `-T` values like `2`**

#### knori

For a help message and to see valid flags:

```
exec/knori
```

An example of how to process with file `datafile.bin`:

```
exec/knori datafile.bin nsamples dim k \
    -t random \
    -T 2 \
    -i 10 \
    -o outdir
```

**For comparison (slower speed)** run our algorithm using
[OpenMP](http://www.openmp.org/) as follows:

```
exec/knori datafile.bin nsamples dim k \
    -t random \
    -T nthread \
    -i 10 \
    -o outdirOMP -O
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

For semi-external memory use, users must configure the underlying user-space filesystem SAFS as described [here](https://github.com/flashxio/FlashX/wiki/FlashX-Quick-Start-Guide)

For a help message:

```
exec/knors
```

An example of how to process with file
`matrix_r50_c5_rrw.adj `:

```
exec/knors libsem/FlashX/flash-graph/conf/run_graph.txt test-data/matrix_r50_c5_rrw.adj 50 5 8
```

The following flags can further aid to accelerate *knors* performance.

```
-r: size of the row cache in gb
```
Specifies another layer within the memory hierarchy to allow for caching at the granularity of rows of data rather than a page like traditional caches. In practice a small cache of 0.5GB-1GB can accelerate the performance of 10 Million+ samples.

```
-I: row cache update interval
```
 The row cache lazily updates users must specify how often to refresh the cache in iterations of the algorithm. Note that as iterations proceed the cache will automatically update less frequently as computation stabilizes.

### Output file from *knor*

The output file obtained by using the `-o` flag within *knor* produces a
[YAML](http://yaml.org/) file that is easily readable by any YAML reader.

For instance, within Python we can use
[PyYAML](https://pypi.python.org/pypi/PyYAML/) which is installable via
by `pip install pyyaml` to read the file as follows:

```python
from yaml import load
with open("kmeans_t.yml", "r") as f:
    kms = load(f) # Returns a python dictionary

kms.keys()
> ['niter', 'dim', 'nsamples', 'cluster', 'k', 'centroids', 'size'
```

The fields are as follows:

- `niter`: The number of iterations performed.
- `dim`: The number of attributes within the dataset.
- `nsamples`: The number of samples within the dataset.
- `cluster`: To which cluster, [0-k), each sample is assigned.
- `k`: The number of clusters requested.
- `centroids`: The `k x dim` centroids matrix with each row
    representing a cluster's centroid.
- `size`: The number of samples placed within each cluster.


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
a leading 4KB FlashGraph header. You can convert binary or text data to *knors* format using the `convert_matrix`
utility.

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
cd $KNOR_HOME
utils/convert_matrix # gives help on how to use the converter

> usage: ./convert_matrix in_filename in_format [text/knori/knord/knors]\
    out_filename out_format[text/knori/knord/knors] nsamples dim

utils/convert_matrix example.txt \
    text \
    example.dat\
    knori\
    2 4
```

## Formatting data directly from other languages

For *knori* and *knord* the format can
be produced by any binary writing utility in any language. **NOTE: ** *knors*
is a custom format that should be generated using the `convert_matrix` utility.
Below are a couple examples of writing *knori* and *knord* format from some
higher-level languages.

### `R`

The following is an example of how to take an R data frame
and write out to raw binary row-major format:

```R
# `mtcars` is a data frame loaded by default into R

mtcars # Take a look at the data

nbytes.per.elem = 8 # NOTE: Always true
mtvec <- as.vector(t(mtcars)) # Turn dataframe into matrix (`t`) then vector
writeBin(mtvec, "yourfilename.bin", size=nbytes.per.elem)
```

### `Python`

The following is an example of how to take a numpy ndarray
and write it out to raw binary row-major format:

```python
import numpy as np
np.random.seed(1) # For consistency
dat = np.random.rand(16, 8)
dat.shape # We have a 16 x 8 matrix

type(dat[0][0]) # Check type is double i.e. 8 bytes = 64 bits
# dat = dat.astype(np.float64) # if data type is wrong

dat.tofile("yourfilename.bin")
```

### Updating *knor*

To update *knor* simply run the update script which pulls directly from
Github:

```
cd $KNOR_HOME
./update.sh
```

### Points of note

- Different *knor* routines **may** not produce the same result
dependent upon which initialization routine is used, **but** every routine
will give the same answer over multiple runs.
- Found a bug? Please [create an issue](https://github.com/flashxio/knor/issues/new) on Github.

## Publications

Mhembere, D., Zheng, D., Vogelstein, J. T., Priebe, C. E., & Burns, R. (2017).
knor: A NUMA-optimized In-memory, Distributed and Semi-external-memory k-means
Library. arXiv preprint arXiv:1606.08905.

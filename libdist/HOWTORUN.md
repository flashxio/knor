## Define
- NP: The number of mpi processes
- EXEC: The exectuable file you want to run
- NT: The number of threads you want to run (either pthreads or OMP)

### Works
mpirun.mpich -n NP --map-by ppr:1:node EXEC [program options]

### Try
mpirun.mpich -n NP --bind-to [none|numa] -report-bindings --map-by ppr:1:node \
            -x OMP_NUM_THREADS=NT

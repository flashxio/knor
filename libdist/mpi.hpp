/*
 * Copyright 2016 neurodata (http://neurodata.io/)
 * Written by Disa Mhembere (disa@jhu.edu)
 *
 * This file is part of k-par-means
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY CURRENT_KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __KPM_MPI__
#define __KPM_MPI__

#include <mpi.h>
#include "exception.hpp"

namespace kpmeans { namespace mpi {
class mpi {
public:
#if 0
    template <typename CType, typename MPIType>
    static void allreduce(const CType* send_buff, const CType* recv_buff,
            const size_t numel=1) {
        int ret = MPI_Allreduce(send_buff, recv_buff,
                numel, MPIType, MPI_SUM, MPI_COMM_WORLD);
        if (ret)
            throw kpmbase::mpi_exception("All reduce failure in reduce_double"
                    , ret);
    }
#endif

    static void reduce_double(const double* send_buff, double* recv_buff,
            const size_t numel=1) {
        int ret = MPI_Allreduce(send_buff, recv_buff,
                numel, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (ret)
            throw kpmbase::mpi_exception("All reduce failure in reduce_double"
                    , ret);
    }

    static void reduce_size_t(const size_t* send_buff, size_t* rev_buff,
            const size_t numel=1) {
        int ret = MPI_Allreduce(send_buff, rev_buff, numel,
                MPI::UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (ret)
            throw kpmbase::mpi_exception("All reduce failure in reduce_size_t"
                    , ret);
    }

    static void allgather_double(const double* send_buff,
            double* recv_buff, const size_t numel) {
        int ret = MPI_Allgather(
                send_buff, numel, MPI_DOUBLE, recv_buff,
                numel, MPI_DOUBLE, MPI_COMM_WORLD);
        if (ret)
            throw kpmbase::mpi_exception("All gather failure in double", ret);
    }

    static void bcast_double(double* buffer, const int pid,
            const size_t numel) {
        int ret = MPI_Bcast(buffer, numel, MPI_DOUBLE, pid, MPI_COMM_WORLD);
        if (ret)
            throw kpmbase::mpi_exception("Bcast failure in double", ret);
    }

    // Merge the per-process cluster assingments so it can be returned in 1 proc
    void merge_global_assignments() { /*FIXME*/ }
};
}} // namespace kpmeans::mpi
#endif

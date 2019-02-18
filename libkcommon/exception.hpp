/*
 * Copyright 2016 neurodata (http://neurodata.io/)
 * Written by Disa Mhembere (disa@jhu.edu)
 *
 * This file is part of knor
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

#ifndef __KNOR_EXCEPTIONS_HPP__
#define __KNOR_EXCEPTIONS_HPP__

#include <exception>
#include <stdexcept>

namespace knor { namespace base {

class not_implemented_exception : public std::runtime_error {

public:
    not_implemented_exception() :
        runtime_error("Method not Implemented!\n") {
        }
};

class abstract_exception : public std::runtime_error {
public:
    abstract_exception() :
        runtime_error("[ERROR]: Cannot call Base class method!\n") {
        }
};

class oob_exception : public std::runtime_error {
public:
    oob_exception(const std::string msg) :
        runtime_error(std::string("[ERROR]: Out of Bounds! ") +
                msg + std::string("\n")) { }
};

class io_exception : public std::runtime_error {
public:
    io_exception(const std::string msg) :
        runtime_error(std::string("[error]: io ") + msg) {
        }

    io_exception(const std::string msg, const int error_code):
        io_exception(msg + std::string(". errcode: ") +
                std::to_string(error_code)) {
    }
};

class parameter_exception : public std::runtime_error {
public:
    parameter_exception(const std::string msg) :
        runtime_error(std::string("parameter error: ") + msg) {
        }

    parameter_exception(const std::string msg, const int error_val):
        parameter_exception(msg + std::string(". Error value: ") +
                std::to_string(error_val)) {
    }

    parameter_exception(const std::string msg, const std::string error_val):
        parameter_exception(msg + std::string(". Error value: ") + error_val) {
    }
};

class mpi_exception : public std::runtime_error {
public:
    mpi_exception(const std::string msg, const int error_code) :
        runtime_error(std::string("[ERROR]: MPI ") + msg +
                ". Error code: " + std::to_string(error_code)) {
        }
};

class thread_exception: public std::exception {

private:
    std::string msg = "[ERROR]: thread_exception ==> ";

    virtual const char* what() const throw() {
        return this->msg.c_str();
    }

public:
    thread_exception(const std::string msg) {
        this->msg += msg;
    }

    thread_exception(const std::string msg, const int rc) :
        thread_exception(msg) {
        this->msg += std::string(". ERRCODE: ") + std::to_string(rc)
            + std::string(" \n");
    }
};
} }
#endif // __KNOR_EXCEPTIONS_HPP__

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// #include <sycl/sycl.hpp>
#include <CL/sycl.hpp>
#include "sum_tree_nary.cpp"

namespace py = pybind11;


PYBIND11_MODULE(sycl_rm_module, m) {
    py::class_<SumTreeNary>(m, "SumTreeNary")
        // .def(py::init<const std::string &>())
        // .def("setName", &Pet::setName)
        .def(py::init<int64_t, int64_t>())
        .def("get_prefix_sum_idx_sycl", &SumTreeNary::get_prefix_sum_idx_sycl)
        .def("set", &SumTreeNary::set);
}

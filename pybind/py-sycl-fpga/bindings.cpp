#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <CL/sycl.hpp>
// #include "rmm.hpp"
#include "replay_cpplib.cpp"

namespace py = pybind11;


PYBIND11_MODULE(replay_module, m) {
    py::class_<PER>(m, "PER")
        // .def(py::init<const std::string &>())
        // .def("setName", &Pet::setName)
        .def(py::init<>())
        .def("Init", &PER::Init_Tree)
        .def("DoWorkMultiKernel", &PER::DoWorkMultiKernel);

    py::class_<sibit_io>(m, "sibit_io")
        .def(py::init<>()) // <-- bind the default constructor
        .def_readwrite("sampling_flag", &sibit_io::sampling_flag)
        .def_readwrite("start", &sibit_io::start)
        .def_readwrite("newx", &sibit_io::newx)
        .def_readwrite("update_index_array", &sibit_io::update_index_array)
        .def_readwrite("update_offset_array", &sibit_io::update_offset_array)
        .def_readwrite("update_flag", &sibit_io::update_flag)
        .def_readwrite("get_priority_flag", &sibit_io::get_priority_flag)
        .def_readwrite("pr_idx", &sibit_io::pr_idx)
        .def_readwrite("init_flag", &sibit_io::init_flag);
        // .def("get_upd_input_index", &sibit_io::get_upd_input_index)
        // .def("get_upd_offset_index", &sibit_io::get_upd_offset_index)
        // .def("set_upd_input_index", &sibit_io::set_upd_input_index)
        // .def("set_upd_offset_index", &sibit_io::set_upd_offset_index);

    py::class_<MultiKernel_out>(m, "MultiKernel_out")
        .def(py::init<int>())
        .def_readwrite("sampled_idx", &MultiKernel_out::sampled_idx)
        .def_readwrite("out_pr_sampled", &MultiKernel_out::out_pr_sampled)
        .def_readwrite("out_pr_insertion", &MultiKernel_out::out_pr_insertion)
        .def_readwrite("root_pr", &MultiKernel_out::root_pr);

}

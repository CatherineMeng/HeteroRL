#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include "MLP_train_sycl.cpp"

namespace py = pybind11;


PYBIND11_MODULE(sycl_learner_module, m) {
    py::class_<DQNTrainer>(m, "DQNTrainer")
        // .def(py::init<const std::string &>())
        // .def("setName", &Pet::setName)
        .def(py::init<std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>, std::vector<float>>())
        .def("train_itr", &DQNTrainer::train_itr)
        .def("updated_params", &DQNTrainer::updated_params);

    py::class_<params_out>(m, "params_out")
        .def(py::init<>()) // <-- bind the default constructor
        .def_readwrite("hiddenWeights_d", &params_out::hiddenWeights_d)
        .def_readwrite("hiddenBiases_d", &params_out::hiddenBiases_d)
        .def_readwrite("outputWeights_d", &params_out::outputWeights_d)
        .def_readwrite("outputBiases_d", &params_out::outputBiases_d)
        .def("print_params", &params_out::print_params);


}

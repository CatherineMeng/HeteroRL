#ifndef __AUTORUN_HPP__
#define __AUTORUN_HPP__

// On devcloud:
// #include <sycl/sycl.hpp>
// On local install:
#include <CL/sycl.hpp>
#include <type_traits>

/*
This header defines the Autorun kernel utility. This utility is used to
launch kernels that are submitted before main begins. It is typically used
to launch kernels that run forever.

Autorun creates an autorun kernel that is NOT implicitly wrapped in an infinite loop.
Usually when using Autorun the user will have the while(1) loop explicitly in their code.
The following describes the common template and constructor arguments for both the Autorun and AutorunForever.
Template Args:
  KernelID (optional): the name of the autorun kernel.
  DeviceSelector: The type of the device selector.
  KernelFunctor: the kernel functor type.
Constructor Arguments:
    device_selector: the SYCL device selector
    kernel: the user-defined kernel functor.
            This defines the logic of the autorun kernel.
*/
namespace fpga_tools {

namespace detail {
// Autorun implementation
template <typename KernelID>
struct Autorun_impl {
  // Constructor with a kernel name
  template <typename DeviceSelector, typename KernelFunctor>
  Autorun_impl(DeviceSelector device_selector, KernelFunctor kernel) {
    // static asserts to ensure KernelFunctor is callable
    static_assert(std::is_invocable_r_v<void, KernelFunctor>,
                  "KernelFunctor must be callable with no arguments");

    // create the device queue
    sycl::queue q{device_selector};

    // submit the user's kernel
      // Autorun: run the kernel as-is, if the user wanted it to run forever they write their own explicit while-loop
      if constexpr (std::is_same_v<KernelID, void>) {
        // Autorun, kernel name not given
        q.single_task(kernel);
      } else {
        // Autorun, kernel name given
        q.single_task<KernelID>(kernel);
      }
    
  }
};
}  // namespace detail

// Autorun
template <typename KernelID = void>
using Autorun = detail::Autorun_impl<false, KernelID>;

// AutorunForever
template <typename KernelID = void>
using AutorunForever = detail::Autorun_impl<true, KernelID>;
}  // namespace fpga_tools

#endif /* __AUTORUN_HPP__ */
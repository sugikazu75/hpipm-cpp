#ifndef HPIPM_CPP_OCP_QP_DIM_WRAPPER_HPP_
#define HPIPM_CPP_OCP_QP_DIM_WRAPPER_HPP_

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "hpipm-cpp/detail/hpipm_traits.hpp"


namespace hpipm {

///
/// @class ocp_qp_dim_wrapper
/// @brief A wrapper of hpipm's ocp_qp_dim with memory management, templated on
/// the scalar type (double or float). The precision-dependent hpipm symbols are
/// dispatched through hpipm_traits<Scalar>.
///
template <typename Scalar>
class ocp_qp_dim_wrapper {
public:
  using traits = hpipm_traits<Scalar>;
  using ocp_qp_dim = typename traits::ocp_qp_dim;

  ///
  /// @brief Constructor. Allocates the hpipm resource.
  /// @param[in] N length of the horizon.
  ///
  ocp_qp_dim_wrapper(const unsigned int N)
    : ocp_qp_dim_wrapper() {
    if (N > 0) {
      resize(N);
    }
  }

  ///
  /// @brief Default constructor. Does not allocate the hpipm resource.
  ///
  ocp_qp_dim_wrapper()
    : ocp_qp_dim_hpipm_(),
      memory_(nullptr),
      memsize_(0) {
  }

  ///
  /// @brief Destructor.
  ///
  ~ocp_qp_dim_wrapper() {
    if (memory_) {
      free(memory_);
      memory_ = nullptr;
      memsize_ = 0;
    }
  }

  ///
  /// @brief Custom copy constructor.
  ///
  ocp_qp_dim_wrapper(const ocp_qp_dim_wrapper& other)
    : ocp_qp_dim_wrapper() {
    copy(other);
  }

  ///
  /// @brief Custom copy assign operator.
  ///
  ocp_qp_dim_wrapper& operator=(const ocp_qp_dim_wrapper& other) {
    if (this != &other) {
      copy(other);
    }
    return *this;
  }

  ///
  /// @brief Custom move constructor.
  ///
  ocp_qp_dim_wrapper(ocp_qp_dim_wrapper&& other) noexcept
    : ocp_qp_dim_hpipm_(other.ocp_qp_dim_hpipm_),
      memory_(other.memory_),
      memsize_(other.memsize_) {
    other.nullify();
  }

  ///
  /// @brief Custom move assign operator.
  ///
  ocp_qp_dim_wrapper& operator=(ocp_qp_dim_wrapper&& other) noexcept {
    if (this == &other) return *this;

    if (memory_) {
      free(memory_);
      memory_ = nullptr;
      memsize_ = 0;
    }
    ocp_qp_dim_hpipm_ = other.ocp_qp_dim_hpipm_;
    memory_ = other.memory_;
    memsize_ = other.memsize_;
    other.nullify();
    return *this;
  }

  ///
  /// @brief Gets the pointer to the hpipm resource. Throw an exception if the
  /// memory for the instance is not allocated.
  /// @return Pointer to the hpipm resource.
  ///
  ocp_qp_dim* get() {
    if (memory_ == nullptr) {
      throw std::runtime_error("[ocp_qp_dim_wrapper] hpipm memory is not created. Call resize() first.");
    }
    return &ocp_qp_dim_hpipm_;
  }

  ///
  /// @brief Gets the const pointer to the hpipm instance.
  /// @return const pointer to the hpipm resource.
  ///
  const ocp_qp_dim* get() const {
    if (memory_ == nullptr) {
      throw std::runtime_error("[ocp_qp_dim_wrapper] hpipm memory is not created. Call resize() first.");
    }
    return &ocp_qp_dim_hpipm_;
  }

  ///
  /// @brief Resizes the hpipm resource.
  /// @param[in] N length of the horizon.
  ///
  void resize(const unsigned int N) {
    const hpipm_size_t new_memsize = traits::ocp_qp_dim_memsize(N);
    if (memory_ != nullptr && new_memsize > memsize_) {
      free(memory_);
      memory_ = nullptr;
    }
    memsize_ = std::max(memsize_, new_memsize);
    if (memory_ == nullptr) {
      memory_ = malloc(memsize_);
    }
    if (ocp_qp_dim_hpipm_.N != static_cast<int>(N)) {
      traits::ocp_qp_dim_create(N, &ocp_qp_dim_hpipm_, memory_);
    }
  }

private:
  ocp_qp_dim ocp_qp_dim_hpipm_;
  void *memory_ = nullptr;
  hpipm_size_t memsize_ = 0;

  void copy(const ocp_qp_dim_wrapper& other) {
    if (memory_) {
      free(memory_);
      memory_ = nullptr;
    }
    memsize_ = 0;
    resize(static_cast<unsigned int>(other.get()->N));

    traits::ocp_qp_dim_copy_all(const_cast<ocp_qp_dim*>(other.get()),
                                &ocp_qp_dim_hpipm_);
  }

  /// @brief Resets the moved-from instance to an empty, owning-nothing state.
  void nullify() {
    ocp_qp_dim_hpipm_.nx   = nullptr;
    ocp_qp_dim_hpipm_.nu   = nullptr;
    ocp_qp_dim_hpipm_.nb   = nullptr;
    ocp_qp_dim_hpipm_.nbx  = nullptr;
    ocp_qp_dim_hpipm_.nbu  = nullptr;
    ocp_qp_dim_hpipm_.ng   = nullptr;
    ocp_qp_dim_hpipm_.ns   = nullptr;
    ocp_qp_dim_hpipm_.nbxe = nullptr;
    ocp_qp_dim_hpipm_.nbue = nullptr;
    ocp_qp_dim_hpipm_.nge  = nullptr;
    ocp_qp_dim_hpipm_.N = 0;
    ocp_qp_dim_hpipm_.memsize = 0;
    memory_ = nullptr;
    memsize_ = 0;
  }
};

} // namespace hpipm

#endif // HPIPM_CPP_OCP_QP_DIM_WRAPPER_HPP_

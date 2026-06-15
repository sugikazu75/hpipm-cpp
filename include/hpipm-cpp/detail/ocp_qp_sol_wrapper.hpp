#ifndef HPIPM_CPP_OCP_QP_SOL_WRAPPER_HPP_
#define HPIPM_CPP_OCP_QP_SOL_WRAPPER_HPP_

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "hpipm-cpp/detail/hpipm_traits.hpp"
#include "hpipm-cpp/detail/ocp_qp_dim_wrapper.hpp"


namespace hpipm {

///
/// @class ocp_qp_sol_wrapper
/// @brief A wrapper of hpipm's ocp_qp_sol with memory management, templated on
/// the scalar type (double or float).
///
template <typename Scalar>
class ocp_qp_sol_wrapper {
public:
  using traits = hpipm_traits<Scalar>;
  using ocp_qp_sol = typename traits::ocp_qp_sol;
  using dim_wrapper = ocp_qp_dim_wrapper<Scalar>;

  ocp_qp_sol_wrapper(const std::shared_ptr<dim_wrapper>& dim)
    : ocp_qp_sol_wrapper() {
    resize(dim);
  }

  ocp_qp_sol_wrapper()
    : dim_(),
      ocp_qp_sol_hpipm_(),
      memory_(nullptr),
      memsize_(0) {
  }

  ~ocp_qp_sol_wrapper() {
    if (memory_) {
      free(memory_);
      memory_ = nullptr;
      memsize_ = 0;
    }
  }

  ocp_qp_sol_wrapper(const ocp_qp_sol_wrapper& other)
    : ocp_qp_sol_wrapper() {
    copy(other);
  }

  ocp_qp_sol_wrapper& operator=(const ocp_qp_sol_wrapper& other) {
    if (this != &other) {
      copy(other);
    }
    return *this;
  }

  ocp_qp_sol_wrapper(ocp_qp_sol_wrapper&& other) noexcept
    : dim_(std::move(other.dim_)),
      ocp_qp_sol_hpipm_(other.ocp_qp_sol_hpipm_),
      memory_(other.memory_),
      memsize_(other.memsize_) {
    other.nullify();
  }

  ocp_qp_sol_wrapper& operator=(ocp_qp_sol_wrapper&& other) noexcept {
    if (this == &other) return *this;

    if (memory_) {
      free(memory_);
      memory_ = nullptr;
      memsize_ = 0;
    }
    dim_ = std::move(other.dim_);
    ocp_qp_sol_hpipm_ = other.ocp_qp_sol_hpipm_;
    memory_ = other.memory_;
    memsize_ = other.memsize_;
    other.nullify();
    return *this;
  }

  ocp_qp_sol* get() {
    if (memory_ == nullptr) {
      throw std::runtime_error("[ocp_qp_sol_wrapper] hpipm memory is not created. Call resize() first.");
    }
    return &ocp_qp_sol_hpipm_;
  }

  const ocp_qp_sol* get() const {
    if (memory_ == nullptr) {
      throw std::runtime_error("[ocp_qp_sol_wrapper] hpipm memory is not created. Call resize() first.");
    }
    return &ocp_qp_sol_hpipm_;
  }

  void resize(const std::shared_ptr<dim_wrapper>& dim) {
    dim_ = dim;
    const hpipm_size_t new_memsize = traits::ocp_qp_sol_memsize(dim_->get());
    if (memory_ != nullptr && new_memsize > memsize_) {
      free(memory_);
      memory_ = nullptr;
    }
    memsize_ = std::max(memsize_, new_memsize);
    if (memory_ == nullptr) {
      memory_ = malloc(memsize_);
    }
    traits::ocp_qp_sol_create(dim_->get(), &ocp_qp_sol_hpipm_, memory_);
  }

private:
  std::shared_ptr<dim_wrapper> dim_;
  ocp_qp_sol ocp_qp_sol_hpipm_;
  void *memory_ = nullptr;
  hpipm_size_t memsize_ = 0;

  void copy(const ocp_qp_sol_wrapper& other) {
    resize(other.dim_);
    traits::ocp_qp_sol_copy_all(const_cast<ocp_qp_sol*>(other.get()),
                                &ocp_qp_sol_hpipm_);
  }

  void nullify() {
    ocp_qp_sol_hpipm_.dim  = nullptr;
    ocp_qp_sol_hpipm_.ux   = nullptr;
    ocp_qp_sol_hpipm_.pi   = nullptr;
    ocp_qp_sol_hpipm_.lam  = nullptr;
    ocp_qp_sol_hpipm_.t    = nullptr;
    ocp_qp_sol_hpipm_.misc = nullptr;
    ocp_qp_sol_hpipm_.memsize = 0;
    memory_ = nullptr;
    memsize_ = 0;
  }
};

} // namespace hpipm

#endif // HPIPM_CPP_OCP_QP_SOL_WRAPPER_HPP_

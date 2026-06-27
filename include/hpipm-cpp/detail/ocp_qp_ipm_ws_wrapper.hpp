#ifndef HPIPM_CPP_OCP_QP_IPM_WS_WRAPPER_HPP_
#define HPIPM_CPP_OCP_QP_IPM_WS_WRAPPER_HPP_

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "hpipm-cpp/detail/hpipm_traits.hpp"
#include "hpipm-cpp/detail/ocp_qp_dim_wrapper.hpp"
#include "hpipm-cpp/detail/ocp_qp_ipm_arg_wrapper.hpp"


namespace hpipm {

///
/// @class ocp_qp_ipm_ws_wrapper
/// @brief A wrapper of hpipm's ocp_qp_ipm_ws with memory management, templated
/// on the scalar type (double or float).
///
template <typename Scalar>
class ocp_qp_ipm_ws_wrapper {
public:
  using traits = hpipm_traits<Scalar>;
  using ocp_qp_ipm_ws = typename traits::ocp_qp_ipm_ws;
  using dim_wrapper = ocp_qp_dim_wrapper<Scalar>;
  using arg_wrapper = ocp_qp_ipm_arg_wrapper<Scalar>;

  ocp_qp_ipm_ws_wrapper(const std::shared_ptr<dim_wrapper>& dim,
                        const std::shared_ptr<arg_wrapper>& ipm_arg)
    : ocp_qp_ipm_ws_wrapper() {
    resize(dim, ipm_arg);
  }

  ocp_qp_ipm_ws_wrapper()
    : dim_(),
      ipm_arg_(),
      ocp_qp_ipm_ws_hpipm_(),
      memory_(nullptr),
      memsize_(0) {
  }

  ~ocp_qp_ipm_ws_wrapper() {
    if (memory_) {
      free(memory_);
      memory_ = nullptr;
      memsize_ = 0;
    }
  }

  ocp_qp_ipm_ws_wrapper(const ocp_qp_ipm_ws_wrapper&) = delete;
  ocp_qp_ipm_ws_wrapper& operator=(const ocp_qp_ipm_ws_wrapper&) = delete;

  ocp_qp_ipm_ws_wrapper(ocp_qp_ipm_ws_wrapper&& other) noexcept
    : dim_(std::move(other.dim_)),
      ipm_arg_(std::move(other.ipm_arg_)),
      ocp_qp_ipm_ws_hpipm_(other.ocp_qp_ipm_ws_hpipm_),
      memory_(other.memory_),
      memsize_(other.memsize_) {
    other.nullify();
  }

  ocp_qp_ipm_ws_wrapper& operator=(ocp_qp_ipm_ws_wrapper&& other) noexcept {
    if (this == &other) return *this;

    if (memory_) {
      free(memory_);
      memory_ = nullptr;
      memsize_ = 0;
    }
    dim_ = std::move(other.dim_);
    ipm_arg_ = std::move(other.ipm_arg_);
    ocp_qp_ipm_ws_hpipm_ = other.ocp_qp_ipm_ws_hpipm_;
    memory_ = other.memory_;
    memsize_ = other.memsize_;
    other.nullify();
    return *this;
  }

  ocp_qp_ipm_ws* get() {
    if (memory_ == nullptr) {
      throw std::runtime_error("[ocp_qp_ipm_ws_wrapper] hpipm memory is not created. Call resize() first.");
    }
    return &ocp_qp_ipm_ws_hpipm_;
  }

  const ocp_qp_ipm_ws* get() const {
    if (memory_ == nullptr) {
      throw std::runtime_error("[ocp_qp_ipm_ws_wrapper] hpipm memory is not created. Call resize() first.");
    }
    return &ocp_qp_ipm_ws_hpipm_;
  }

  void resize(const std::shared_ptr<dim_wrapper>& dim,
              const std::shared_ptr<arg_wrapper>& ipm_arg) {
    dim_ = dim;
    ipm_arg_ = ipm_arg;
    ipm_arg_->resize(dim_);
    const hpipm_size_t new_memsize =
        traits::ocp_qp_ipm_ws_memsize(dim_->get(), ipm_arg_->get());
    if (memory_ != nullptr && new_memsize > memsize_) {
      free(memory_);
      memory_ = nullptr;
    }
    memsize_ = std::max(memsize_, new_memsize);
    if (memory_ == nullptr) {
      memory_ = malloc(memsize_);
    }
    traits::ocp_qp_ipm_ws_create(dim_->get(), ipm_arg_->get(),
                                 &ocp_qp_ipm_ws_hpipm_, memory_);
  }

private:
  std::shared_ptr<dim_wrapper> dim_;
  std::shared_ptr<arg_wrapper> ipm_arg_;
  ocp_qp_ipm_ws ocp_qp_ipm_ws_hpipm_;
  void *memory_ = nullptr;
  hpipm_size_t memsize_ = 0;

  void nullify() {
    ocp_qp_ipm_ws_hpipm_.core_workspace = nullptr;
    ocp_qp_ipm_ws_hpipm_.dim = nullptr;
    ocp_qp_ipm_ws_hpipm_.res_workspace = nullptr;
    ocp_qp_ipm_ws_hpipm_.sol_step = nullptr;
    ocp_qp_ipm_ws_hpipm_.sol_itref = nullptr;
    ocp_qp_ipm_ws_hpipm_.qp_step = nullptr;
    ocp_qp_ipm_ws_hpipm_.qp_itref = nullptr;
    ocp_qp_ipm_ws_hpipm_.res_itref = nullptr;
    ocp_qp_ipm_ws_hpipm_.res = nullptr;
    ocp_qp_ipm_ws_hpipm_.Gamma = nullptr;
    ocp_qp_ipm_ws_hpipm_.gamma = nullptr;
    ocp_qp_ipm_ws_hpipm_.tmp_nuxM = nullptr;
    ocp_qp_ipm_ws_hpipm_.tmp_nbgM = nullptr;
    ocp_qp_ipm_ws_hpipm_.Pb = nullptr;
    ocp_qp_ipm_ws_hpipm_.Zs_inv = nullptr;
    ocp_qp_ipm_ws_hpipm_.tmp_m = nullptr;
    ocp_qp_ipm_ws_hpipm_.l = nullptr;
    ocp_qp_ipm_ws_hpipm_.L = nullptr;
    ocp_qp_ipm_ws_hpipm_.Ls = nullptr;
    ocp_qp_ipm_ws_hpipm_.P = nullptr;
    ocp_qp_ipm_ws_hpipm_.Lh = nullptr;
    ocp_qp_ipm_ws_hpipm_.AL = nullptr;
    ocp_qp_ipm_ws_hpipm_.lq0 = nullptr;
    ocp_qp_ipm_ws_hpipm_.tmp_nxM_nxM = nullptr;
    ocp_qp_ipm_ws_hpipm_.stat = nullptr;
    ocp_qp_ipm_ws_hpipm_.use_hess_fact = nullptr;
    ocp_qp_ipm_ws_hpipm_.lq_work0 = nullptr;
    ocp_qp_ipm_ws_hpipm_.memsize = 0;
    memory_ = nullptr;
    memsize_ = 0;
  }
};

} // namespace hpipm

#endif // HPIPM_CPP_OCP_QP_IPM_WS_WRAPPER_HPP_

#ifndef HPIPM_CPP_OCP_QP_IPM_ARG_WRAPPER_HPP_
#define HPIPM_CPP_OCP_QP_IPM_ARG_WRAPPER_HPP_

#include <cstdlib>
#include <memory>

#include "hpipm-cpp/detail/hpipm_traits.hpp"
#include "hpipm-cpp/detail/ocp_qp_dim_wrapper.hpp"


namespace hpipm {

///
/// @class ocp_qp_ipm_arg_wrapper
/// @brief A wrapper of hpipm's ocp_qp_ipm_arg with memory management, templated
/// on the scalar type (double or float).
///
template <typename Scalar>
class ocp_qp_ipm_arg_wrapper {
public:
  using traits = hpipm_traits<Scalar>;
  using ocp_qp_dim = typename traits::ocp_qp_dim;
  using ocp_qp_ipm_arg = typename traits::ocp_qp_ipm_arg;
  using dim_wrapper = ocp_qp_dim_wrapper<Scalar>;

  ocp_qp_ipm_arg_wrapper()
    : ocp_qp_ipm_arg_hpipm_(),
      memory_(nullptr),
      memsize_(0) {
    resize();
  }

  ~ocp_qp_ipm_arg_wrapper() {
    if (memory_) {
      free(memory_);
      memory_ = nullptr;
      memsize_ = 0;
    }
  }

  ocp_qp_ipm_arg_wrapper(const ocp_qp_ipm_arg_wrapper& other)
    : ocp_qp_ipm_arg_wrapper() {
    copy(other);
  }

  ocp_qp_ipm_arg_wrapper& operator=(const ocp_qp_ipm_arg_wrapper& other) {
    if (this != &other) {
      copy(other);
    }
    return *this;
  }

  ocp_qp_ipm_arg_wrapper(ocp_qp_ipm_arg_wrapper&& other) noexcept
    : ocp_qp_ipm_arg_hpipm_(other.ocp_qp_ipm_arg_hpipm_),
      memory_(other.memory_),
      memsize_(other.memsize_) {
    other.memory_ = nullptr;
    other.memsize_ = 0;
  }

  ocp_qp_ipm_arg_wrapper& operator=(ocp_qp_ipm_arg_wrapper&& other) noexcept {
    if (this == &other) return *this;

    if (memory_) {
      free(memory_);
      memory_ = nullptr;
      memsize_ = 0;
    }
    ocp_qp_ipm_arg_hpipm_ = other.ocp_qp_ipm_arg_hpipm_;
    memory_ = other.memory_;
    memsize_ = other.memsize_;

    other.memory_ = nullptr;
    other.memsize_ = 0;
    return *this;
  }

  ocp_qp_ipm_arg* get() {
    return &ocp_qp_ipm_arg_hpipm_;
  }

  const ocp_qp_ipm_arg* get() const {
    return &ocp_qp_ipm_arg_hpipm_;
  }

  void resize(const std::shared_ptr<dim_wrapper>& dim) {
    resize_with_dim(dim->get());
  }

private:
  ocp_qp_ipm_arg ocp_qp_ipm_arg_hpipm_;
  void *memory_ = nullptr;
  hpipm_size_t memsize_ = 0;

  void resize() {
    dim_wrapper dim(1);
    resize_with_dim(dim.get());
  }

  void resize_with_dim(const ocp_qp_dim* dim) {
    const hpipm_size_t new_memsize =
        traits::ocp_qp_ipm_arg_memsize(const_cast<ocp_qp_dim*>(dim));
    if (memory_ && new_memsize != memsize_) {
      free(memory_);
      memory_ = nullptr;
    }
    memsize_ = new_memsize;
    if (!memory_) {
      memory_ = malloc(memsize_);
    }
    traits::ocp_qp_ipm_arg_create(const_cast<ocp_qp_dim*>(dim),
                                  &ocp_qp_ipm_arg_hpipm_, memory_);
    traits::ocp_qp_ipm_arg_set_default(hpipm_mode::SPEED, &ocp_qp_ipm_arg_hpipm_);
  }

  void copy(const ocp_qp_ipm_arg_wrapper& other) {
    const ocp_qp_ipm_arg* other_ptr = other.get();
    ocp_qp_ipm_arg_hpipm_.mu0 = other_ptr->mu0;
    ocp_qp_ipm_arg_hpipm_.alpha_min = other_ptr->alpha_min;
    ocp_qp_ipm_arg_hpipm_.res_g_max = other_ptr->res_g_max;
    ocp_qp_ipm_arg_hpipm_.res_b_max = other_ptr->res_b_max;
    ocp_qp_ipm_arg_hpipm_.res_d_max = other_ptr->res_d_max;
    ocp_qp_ipm_arg_hpipm_.res_m_max = other_ptr->res_m_max;
    ocp_qp_ipm_arg_hpipm_.reg_prim = other_ptr->reg_prim;
    ocp_qp_ipm_arg_hpipm_.lam_min = other_ptr->lam_min;
    ocp_qp_ipm_arg_hpipm_.t_min = other_ptr->t_min;
    ocp_qp_ipm_arg_hpipm_.tau_min = other_ptr->tau_min;
    ocp_qp_ipm_arg_hpipm_.iter_max = other_ptr->iter_max;
    ocp_qp_ipm_arg_hpipm_.stat_max = other_ptr->stat_max;
    ocp_qp_ipm_arg_hpipm_.pred_corr = other_ptr->pred_corr;
    ocp_qp_ipm_arg_hpipm_.cond_pred_corr = other_ptr->cond_pred_corr;
    ocp_qp_ipm_arg_hpipm_.itref_pred_max = other_ptr->itref_pred_max;
    ocp_qp_ipm_arg_hpipm_.itref_corr_max = other_ptr->itref_corr_max;
    ocp_qp_ipm_arg_hpipm_.warm_start = other_ptr->warm_start;
    ocp_qp_ipm_arg_hpipm_.square_root_alg = other_ptr->square_root_alg;
    ocp_qp_ipm_arg_hpipm_.lq_fact = other_ptr->lq_fact;
    ocp_qp_ipm_arg_hpipm_.abs_form = other_ptr->abs_form;
    ocp_qp_ipm_arg_hpipm_.comp_dual_sol_eq = other_ptr->comp_dual_sol_eq;
    ocp_qp_ipm_arg_hpipm_.comp_res_exit = other_ptr->comp_res_exit;
    ocp_qp_ipm_arg_hpipm_.comp_res_pred = other_ptr->comp_res_pred;
    ocp_qp_ipm_arg_hpipm_.split_step = other_ptr->split_step;
    ocp_qp_ipm_arg_hpipm_.var_init_scheme = other_ptr->var_init_scheme;
    ocp_qp_ipm_arg_hpipm_.t_lam_min = other_ptr->t_lam_min;
  }
};

} // namespace hpipm

#endif // HPIPM_CPP_OCP_QP_IPM_ARG_WRAPPER_HPP_

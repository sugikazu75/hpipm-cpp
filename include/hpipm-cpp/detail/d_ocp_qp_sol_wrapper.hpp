#ifndef HPIPM_CPP_D_OCP_QP_SOL_WRAPPER_HPP_
#define HPIPM_CPP_D_OCP_QP_SOL_WRAPPER_HPP_

#include "hpipm-cpp/detail/ocp_qp_sol_wrapper.hpp"
#include "hpipm-cpp/detail/d_ocp_qp_dim_wrapper.hpp"


namespace hpipm {

/// @brief Backward-compatible alias for the double-precision sol wrapper.
using d_ocp_qp_sol_wrapper = ocp_qp_sol_wrapper<double>;

} // namespace hpipm

#endif // HPIPM_CPP_D_OCP_QP_SOL_WRAPPER_HPP_

#ifndef HPIPM_CPP_D_OCP_QP_IPM_WS_WRAPPER_HPP_
#define HPIPM_CPP_D_OCP_QP_IPM_WS_WRAPPER_HPP_

#include "hpipm-cpp/detail/ocp_qp_ipm_ws_wrapper.hpp"
#include "hpipm-cpp/detail/d_ocp_qp_dim_wrapper.hpp"
#include "hpipm-cpp/detail/d_ocp_qp_ipm_arg_wrapper.hpp"


namespace hpipm {

/// @brief Backward-compatible alias for the double-precision ipm-ws wrapper.
using d_ocp_qp_ipm_ws_wrapper = ocp_qp_ipm_ws_wrapper<double>;

} // namespace hpipm

#endif // HPIPM_CPP_D_OCP_QP_IPM_WS_WRAPPER_HPP_

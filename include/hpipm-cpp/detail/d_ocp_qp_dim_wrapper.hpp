#ifndef HPIPM_CPP_D_OCP_QP_DIM_WRAPPER_HPP_
#define HPIPM_CPP_D_OCP_QP_DIM_WRAPPER_HPP_

#include "hpipm-cpp/detail/ocp_qp_dim_wrapper.hpp"


namespace hpipm {

///
/// @brief Backward-compatible alias for the double-precision dim wrapper.
/// New code should prefer ocp_qp_dim_wrapper<Scalar> directly.
///
using d_ocp_qp_dim_wrapper = ocp_qp_dim_wrapper<double>;

} // namespace hpipm

#endif // HPIPM_CPP_D_OCP_QP_DIM_WRAPPER_HPP_

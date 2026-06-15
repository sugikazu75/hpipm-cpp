#include "hpipm-cpp/detail/ocp_qp_ipm_arg_wrapper.hpp"


namespace hpipm {

// Header-only template; explicitly instantiate both precisions so the symbols
// are emitted into the shared library.
template class ocp_qp_ipm_arg_wrapper<double>;
template class ocp_qp_ipm_arg_wrapper<float>;

} // namespace hpipm

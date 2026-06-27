#include "hpipm-cpp/detail/ocp_qp_ipm_ws_wrapper.hpp"


namespace hpipm {

// Header-only template; explicitly instantiate both precisions so the symbols
// are emitted into the shared library.
template class ocp_qp_ipm_ws_wrapper<double>;
template class ocp_qp_ipm_ws_wrapper<float>;

} // namespace hpipm

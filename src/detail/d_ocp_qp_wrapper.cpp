#include "hpipm-cpp/detail/ocp_qp_wrapper.hpp"


namespace hpipm {

// Header-only template; explicitly instantiate both precisions so the symbols
// are emitted into the shared library.
template class ocp_qp_wrapper<double>;
template class ocp_qp_wrapper<float>;

} // namespace hpipm

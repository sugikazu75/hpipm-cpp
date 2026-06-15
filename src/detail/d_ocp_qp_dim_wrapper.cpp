#include "hpipm-cpp/detail/ocp_qp_dim_wrapper.hpp"


namespace hpipm {

// The dim wrapper is now a header-only template. Explicitly instantiate both
// precisions so the symbols are emitted into the shared library.
template class ocp_qp_dim_wrapper<double>;
template class ocp_qp_dim_wrapper<float>;

} // namespace hpipm

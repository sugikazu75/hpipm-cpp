#include "hpipm-cpp/ocp_qp_dim.hpp"


namespace hpipm {

OcpQpDim::OcpQpDim(const unsigned int N) {
  resize(N);
}


void OcpQpDim::resize(const unsigned int _N) {
  N = _N;
  nx.resize(N+1);
  nu.resize(N+1);
  nbx.resize(N+1);
  nbu.resize(N+1);
  ng.resize(N+1);
  nsbx.resize(N+1);
  nsbu.resize(N+1);
  nsg.resize(N+1);
}

// The OCP-QP-data-taking overloads (constructor, resize, checkSize) are
// templated on the scalar type and defined in ocp_qp_dim.hxx.

} // namespace hpipm

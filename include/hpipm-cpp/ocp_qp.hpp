#ifndef HPIPM_CPP_OCP_QP_HPP_
#define HPIPM_CPP_OCP_QP_HPP_

#include <vector>

#include "Eigen/Core"


namespace hpipm {

///
/// @class OcpQpTpl
/// @brief The OCP-QP data class, templated on the scalar type (double / float).
///
template <typename Scalar>
struct OcpQpTpl {
  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  ///
  /// @brief Dynamics matrix in x[i+1] = A[i] x[i] + B[i] u[i] + b[i].
  ///
  MatrixX A;

  ///
  /// @brief Dynamics matrix in x[i+1] = A[i] x[i] + B[i] u[i] + b[i].
  ///
  MatrixX B;

  ///
  /// @brief Dynamics vector in x[i+1] = A[i] x[i] + B[i] u[i] + b[i].
  ///
  VectorX b;

  ///
  /// @brief Cost matrix in (1/2) * x[i]^T Q[i] x[i] + u[i]^T S[i] x[i] + (1/2) * u[i]^T R[i] u[i] + q[i]^T x[i] + r[i]^T u[i].
  ///
  MatrixX Q;

  ///
  /// @brief Cost matrix in (1/2) * x[i]^T Q[i] x[i] + u[i]^T S[i] x[i] + (1/2) * u[i]^T R[i] u[i] + q[i]^T x[i] + r[i]^T u[i].
  ///
  MatrixX S;

  ///
  /// @brief Cost matrix in (1/2) * x[i]^T Q[i] x[i] + u[i]^T S[i] x[i] + (1/2) * u[i]^T R[i] u[i] + q[i]^T x[i] + r[i]^T u[i].
  ///
  MatrixX R;

  ///
  /// @brief Cost vector in (1/2) * x[i]^T Q[i] x[i] + u[i]^T S[i] x[i] + (1/2) * u[i]^T R[i] u[i] + q[i]^T x[i] + r[i]^T u[i].
  ///
  VectorX q;

  ///
  /// @brief Cost vector in (1/2) * x[i]^T Q[i] x[i] + u[i]^T S[i] x[i] + (1/2) * u[i]^T R[i] u[i] + q[i]^T x[i] + r[i]^T u[i].
  ///
  VectorX r;

  ///
  /// @brief Indices of box constrainted elements of x.
  ///
  std::vector<int> idxbx;

  ///
  /// @brief Lower bounds of box constraints on x.
  ///
  VectorX lbx;

  ///
  /// @brief Upper bounds of box constraints on x.
  /// Size must be OcpQpDim::N+1.
  /// Size of each element must be OcpQpDim::nbx[i].
  ///
  VectorX ubx;

  ///
  /// @brief Masks on the lower bounds of box constraints on x.
  /// Each element must be composed only by 0 or 1.0.
  ///
  VectorX lbx_mask;

  ///
  /// @brief Masks on the upper bounds of box constraints on x.
  /// Each element must be composed only by 0 or 1.0.
  ///
  VectorX ubx_mask;

  ///
  /// @brief Indices of box constrainted elements of u.
  ///
  std::vector<int> idxbu;

  ///
  /// @brief Lower bounds of box constraints on u.
  ///
  VectorX lbu;

  ///
  /// @brief Upper bounds of box constraints on u.
  ///
  VectorX ubu;

  ///
  /// @brief Masks on the lower bounds of box constraints on u.
  ///
  VectorX lbu_mask;

  ///
  /// @brief Masks on the upper bounds of box constraints on u.
  /// Each element must be composed only by 0 or 1.0.
  ///
  VectorX ubu_mask;

  ///
  /// @brief Constraint matrix in lg[i] < C[i] x[i] + D[i] u[i] < ug[i].
  ///
  MatrixX C;

  ///
  /// @brief Constraint matrix in lg[i] < C[i] x[i] + D[i] u[i] < ug[i].
  ///
  MatrixX D;

  ///
  /// @brief Constraint vector in lg[i] < C[i] x[i] + D[i] u[i] < ug[i].
  ///
  VectorX lg;

  ///
  /// @brief Constraint vector in lg[i] < C[i] x[i] + D[i] u[i] < ug[i].
  ///
  VectorX ug;

  ///
  /// @brief Masks on lower bounds in lg[i] < C[i] x[i] + D[i] u[i] < ug[i].
  /// Each element must be composed only by 0 or 1.0.
  ///
  VectorX lg_mask;

  ///
  /// @brief Masks on upper bounds in lg[i] < C[i] x[i] + D[i] u[i] < ug[i].
  /// Each element must be composed only by 0 or 1.0.
  ///
  VectorX ug_mask;

  ///
  /// @brief Matrix in the slack penalty (1/2) sl^T Zl sl + zl^T sl + (1/2) su^T Zu su + zu^T su.
  ///
  MatrixX Zl;

  ///
  /// @brief Matrix in the slack penalty (1/2) sl^T Zl sl + zl^T sl + (1/2) su^T Zu su + zu^T su.
  ///
  MatrixX Zu;

  ///
  /// @brief Vector in the slack penalty (1/2) sl^T Zl sl + zl^T sl + (1/2) su^T Zu su + zu^T su.
  ///
  VectorX zl;

  ///
  /// @brief Vector in the slack penalty (1/2) sl^T Zl sl + zl^T sl + (1/2) su^T Zu su + zu^T su.
  ///
  VectorX zu;

  ///
  /// @brief Indices of box constrainted elements of slack variables.
  ///
  std::vector<int> idxs;

  ///
  /// @brief Lower bounds of box constraints of slack variables.
  ///
  VectorX lls;

  ///
  /// @brief Upper bounds of box constraints of slack variables.
  ///
  VectorX lus;
};

///
/// @brief Backward-compatible alias for the double-precision OCP-QP data.
///
using OcpQp = OcpQpTpl<double>;

} // namespace hpipm

#endif // HPIPM_CPP_OCP_QP_HPP_

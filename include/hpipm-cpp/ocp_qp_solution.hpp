#ifndef HPIPM_CPP_OCP_QP_SOLUTION_HPP_
#define HPIPM_CPP_OCP_QP_SOLUTION_HPP_

#include "Eigen/Core"

namespace hpipm {

///
/// @class OcpQpSolutionTpl
/// @brief Solution of the OCP-QP problem, templated on the scalar type.
///
template <typename Scalar>
struct OcpQpSolutionTpl {
  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  ///
  /// @brief State.
  ///
  VectorX x;

  ///
  /// @brief Control input.
  ///
  VectorX u;

  ///
  /// @brief Costate (the Lagrange multiplier w.r.t the state equation).
  ///
  VectorX pi;

  ///
  /// @brief Riccati matrix P.
  ///
  MatrixX P;

  ///
  /// @brief Riccati vector s.
  ///
  VectorX p;

  ///
  /// @brief Feedback gain.
  ///
  MatrixX K;

  ///
  /// @brief Feedforward term.
  ///
  VectorX k;
};

///
/// @brief Backward-compatible alias for the double-precision OCP-QP solution.
///
using OcpQpSolution = OcpQpSolutionTpl<double>;

} // namespace hpipm

#endif // HPIPM_CPP_OCP_QP_SOLUTION_HPP_

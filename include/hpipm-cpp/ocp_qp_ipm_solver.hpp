#ifndef HPIPM_CPP_OCP_QP_IPM_SOLVER_HPP_
#define HPIPM_CPP_OCP_QP_IPM_SOLVER_HPP_

#include <vector>
#include <memory>
#include <string>
#include <iostream>

#include "Eigen/Core"

#include "hpipm-cpp/ocp_qp.hpp"
#include "hpipm-cpp/ocp_qp_dim.hpp"
#include "hpipm-cpp/ocp_qp_solution.hpp"
#include "hpipm-cpp/ocp_qp_ipm_solver_settings.hpp"
#include "hpipm-cpp/ocp_qp_ipm_solver_statistics.hpp"


namespace hpipm {

///
/// @class HpipmStatus
/// @brief Solver status.
///
enum class HpipmStatus {
  Success = 0,
  MaxIterReached = 1,
  MinStepLengthReached = 2,
  NaNDetected = 3,
  UnknownFailure = 4,
};

std::string to_string(const HpipmStatus& hpipm_status);

std::ostream& operator<<(std::ostream& os, const HpipmStatus& hpipm_status);

///
/// @class OcpQpIpmSolverTpl
/// @brief Ipm solver, templated on the scalar type (double / float).
///
template <typename Scalar>
class OcpQpIpmSolverTpl {
public:
  using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using OcpQp = OcpQpTpl<Scalar>;
  using OcpQpSolution = OcpQpSolutionTpl<Scalar>;

  ///
  /// @brief Constructor.
  /// @param[in] ocp_qp OCP-QP problem.
  /// @param[in] solver_settings Solver settings.
  ///
  OcpQpIpmSolverTpl(const std::vector<OcpQp>& ocp_qp,
                    const OcpQpIpmSolverSettings& solver_settings=OcpQpIpmSolverSettings());

  ///
  /// @brief Constructor.
  /// @param[in] solver_settings Solver settings.
  ///
  OcpQpIpmSolverTpl(const OcpQpIpmSolverSettings& solver_settings=OcpQpIpmSolverSettings());

  ///
  /// @brief Destructor.
  ///
  ~OcpQpIpmSolverTpl();

  ///
  /// @brief Prohibit copy constructor.
  ///
  OcpQpIpmSolverTpl(const OcpQpIpmSolverTpl&) = delete;

  ///
  /// @brief Prohibit copy assign operator.
  ///
  OcpQpIpmSolverTpl& operator=(const OcpQpIpmSolverTpl&) = delete;

  ///
  /// @brief Default move constructor.
  ///
  OcpQpIpmSolverTpl(OcpQpIpmSolverTpl&&) noexcept = default;

  ///
  /// @brief Default move assign operator.
  ///
  OcpQpIpmSolverTpl& operator=(OcpQpIpmSolverTpl&&) noexcept = default;

  ///
  /// @brief Sets the Ipm solver settings.
  /// @param[in] solver_settings Solver settings.
  ///
  void setSolverSettings(const OcpQpIpmSolverSettings& solver_settings);

  ///
  /// @brief Resizes the solver.
  /// @param[in] ocp_qp OCP-QP problem.
  ///
  void resize(const std::vector<OcpQp>& ocp_qp);

  ///
  /// @brief Solves the OCP-QP problem.
  /// @param[in] x0 Initial state.
  /// @param[in] ocp_qp OCP-QP problem.
  /// @param[out] qp_sol Solution of the OCP-QP problem.
  /// @return Solver status.
  ///
  HpipmStatus solve(const VectorX& x0, std::vector<OcpQp>& ocp_qp,
                    std::vector<OcpQpSolution>& qp_sol);

  ///
  /// @brief Get the Ipm solver settings.
  /// @return const reference to the Ipm solver settings.
  ///
  const OcpQpIpmSolverSettings& getIpmSolverSettings() const;

  ///
  /// @brief Get the solver statistics.
  /// @return const reference to the solver statistics.
  ///
  const OcpQpIpmSolverStatistics& getSolverStatistics() const;

private:
  OcpQpIpmSolverSettings solver_settings_;
  OcpQpIpmSolverStatistics solver_statistics_;
  OcpQpDim dim_;

  struct WrapperHolder; // Pimpl
  std::unique_ptr<WrapperHolder> wrapper_holder_;

  // raw pointer storage
  std::vector<Scalar*> A_ptr_;
  std::vector<Scalar*> B_ptr_;
  std::vector<Scalar*> b_ptr_;
  std::vector<Scalar*> Q_ptr_;
  std::vector<Scalar*> S_ptr_;
  std::vector<Scalar*> R_ptr_;
  std::vector<Scalar*> q_ptr_;
  std::vector<Scalar*> r_ptr_;
  std::vector<int*> idxbx_ptr_;
  std::vector<Scalar*> lbx_ptr_;
  std::vector<Scalar*> ubx_ptr_;
  std::vector<Scalar*> lbx_mask_ptr_;
  std::vector<Scalar*> ubx_mask_ptr_;
  std::vector<int*> idxbu_ptr_;
  std::vector<Scalar*> lbu_ptr_;
  std::vector<Scalar*> ubu_ptr_;
  std::vector<Scalar*> lbu_mask_ptr_;
  std::vector<Scalar*> ubu_mask_ptr_;
  std::vector<Scalar*> C_ptr_;
  std::vector<Scalar*> D_ptr_;
  std::vector<Scalar*> lg_ptr_;
  std::vector<Scalar*> ug_ptr_;
  std::vector<Scalar*> lg_mask_ptr_;
  std::vector<Scalar*> ug_mask_ptr_;
  std::vector<Scalar*> Zl_ptr_;
  std::vector<Scalar*> Zu_ptr_;
  std::vector<Scalar*> zl_ptr_;
  std::vector<Scalar*> zu_ptr_;
  std::vector<int*> idxs_ptr_;
  std::vector<Scalar*> lls_ptr_;
  std::vector<Scalar*> lus_ptr_;

  // initial state embedding
  VectorX b0_, r0_;
  MatrixX Lr0_, Lr0_inv_, G0_inv_, H0_, B0t_P1_, A0t_P1_;
};

///
/// @brief Backward-compatible alias for the double-precision IPM solver.
///
using OcpQpIpmSolver = OcpQpIpmSolverTpl<double>;

} // namespace hpipm

#endif // HPIPM_CPP_OCP_QP_IPM_SOLVER_HPP_

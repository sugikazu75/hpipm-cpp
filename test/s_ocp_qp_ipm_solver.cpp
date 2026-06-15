#include <gtest/gtest.h>

#include "hpipm-cpp/hpipm-cpp.hpp"

#include "Eigen/LU"


// End-to-end smoke test of the single-precision (float) IPM solver path. It
// builds a small unconstrained LQR problem in float, solves it via
// OcpQpIpmSolverTpl<float>, and checks the result against a float Riccati
// reference. This exercises the s_* hpipm symbols through the templated stack.
namespace hpipm {

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;

class s_OcpQpIpmSolver_test : public ::testing::Test {
protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};


TEST_F(s_OcpQpIpmSolver_test, unconstrained) {
  const int nx = 5;
  const int nu = 3;
  const unsigned int N = 20;

  std::vector<OcpQpTpl<float>> qp(N+1);
  for (int i=0; i<N; ++i) {
    qp[i].A = MatrixXf::Random(nx, nx);
    qp[i].B = MatrixXf::Random(nx, nu);
    qp[i].b = VectorXf::Random(nx);
  }
  for (int i=0; i<N; ++i) {
    const MatrixXf H = MatrixXf::Random(nx+nu, nx+nu);
    const MatrixXf HH = H * H.transpose();
    qp[i].Q = HH.bottomRightCorner(nx, nx);
    qp[i].S = HH.topRightCorner(nu, nx);
    qp[i].R = HH.topLeftCorner(nu, nu);
    qp[i].R.diagonal() += VectorXf::Random(nu).cwiseAbs().matrix();
    qp[i].q = VectorXf::Random(nx);
    qp[i].r = VectorXf::Random(nu);
  }
  const MatrixXf H = MatrixXf::Random(nx, nx);
  qp[N].Q = H * H.transpose();
  qp[N].q = VectorXf::Random(nx);

  const VectorXf x0 = VectorXf::Random(nx);

  OcpQpIpmSolverSettings solver_settings;
  solver_settings.mode = HpipmMode::Balance;
  // Relax tolerances to the single-precision range.
  solver_settings.tol_stat = 1e-5;
  solver_settings.tol_eq   = 1e-5;
  solver_settings.tol_ineq = 1e-5;
  solver_settings.tol_comp = 1e-5;
  solver_settings.reg_prim = 1e-7;

  std::vector<OcpQpSolutionTpl<float>> solution(N+1);
  OcpQpIpmSolverTpl<float> solver(qp, solver_settings);
  const auto status = solver.solve(x0, qp, solution);
  EXPECT_EQ(status, HpipmStatus::Success);
  EXPECT_TRUE(solution[0].x.isApprox(x0));

  // Reference unconstrained Riccati recursion (in float).
  std::vector<MatrixXf> P(N+1), K(N);
  std::vector<VectorXf> s(N+1), k(N);
  P[N] = qp[N].Q;
  s[N] = - qp[N].q;
  for (int i=N-1; i>=0; --i) {
    const MatrixXf F = qp[i].Q + qp[i].A.transpose() * P[i+1] * qp[i].A;
    const MatrixXf Hi = qp[i].S + qp[i].B.transpose() * P[i+1] * qp[i].A;
    const MatrixXf G = qp[i].R + qp[i].B.transpose() * P[i+1] * qp[i].B;
    const MatrixXf Ginv = G.inverse();
    K[i] = - Ginv * Hi;
    k[i] = - Ginv * (qp[i].B.transpose() * P[i+1] * qp[i].b - qp[i].B.transpose() * s[i+1] + qp[i].r);
    P[i] = F - K[i].transpose() * G * K[i];
    s[i] = qp[i].A.transpose() * (s[i+1] - P[i+1] * qp[i].b) - qp[i].q - Hi.transpose() * k[i];
  }
  std::vector<VectorXf> x(N+1), u(N);
  x[0] = x0;
  for (int i=0; i<N; ++i) {
    u[i] = K[i] * x[i] + k[i];
    x[i+1] = qp[i].A * x[i] + qp[i].B * u[i] + qp[i].b;
  }

  // Single precision + IPM: compare with a loose tolerance.
  const float prec = 1.0e-2;
  for (int i=0; i<N; ++i) {
    EXPECT_TRUE(u[i].isApprox(solution[i].u, prec))
        << "u[" << i << "] mismatch";
  }
}

// Reproduces the controller situation: controls have no real bounds, but the
// box constraints are still registered with a huge "no bound" sentinel. We try
// (a) the sentinel approach and (b) hpipm bound masks, printing the status of
// each so we can see which one survives single precision.
TEST_F(s_OcpQpIpmSolver_test, inactive_box_bounds) {
  const int nx = 5;
  const int nu = 3;
  const unsigned int N = 20;

  auto build = []() {
    std::vector<OcpQpTpl<float>> qp(N+1);
    for (int i=0; i<N; ++i) {
      qp[i].A = MatrixXf::Random(nx, nx);
      qp[i].B = MatrixXf::Random(nx, nu);
      qp[i].b = VectorXf::Random(nx);
    }
    for (int i=0; i<N; ++i) {
      const MatrixXf H = MatrixXf::Random(nx+nu, nx+nu);
      const MatrixXf HH = H * H.transpose();
      qp[i].Q = HH.bottomRightCorner(nx, nx);
      qp[i].S = HH.topRightCorner(nu, nx);
      qp[i].R = HH.topLeftCorner(nu, nu);
      qp[i].R.diagonal() += VectorXf::Random(nu).cwiseAbs().matrix();
      qp[i].q = VectorXf::Random(nx);
      qp[i].r = VectorXf::Random(nu);
    }
    const MatrixXf H = MatrixXf::Random(nx, nx);
    qp[N].Q = H * H.transpose();
    qp[N].q = VectorXf::Random(nx);
    return qp;
  };

  const VectorXf x0 = VectorXf::Random(nx);

  OcpQpIpmSolverSettings settings;
  settings.mode = HpipmMode::Balance;
  settings.tol_stat = 1e-3; settings.tol_eq = 1e-3;
  settings.tol_ineq = 1e-3; settings.tol_comp = 1e-3;
  settings.reg_prim = 1e-4;

  const float big = 1e9;

  // (a) huge-sentinel bounds (what crocoddyl currently does)
  {
    auto qp = build();
    for (int i=0; i<N; ++i) {
      qp[i].idxbu = {0, 1, 2};
      qp[i].lbu = VectorXf::Constant(nu, -big);
      qp[i].ubu = VectorXf::Constant(nu,  big);
    }
    std::vector<OcpQpSolutionTpl<float>> sol(N+1);
    OcpQpIpmSolverTpl<float> solver(qp, settings);
    const auto status = solver.solve(x0, qp, sol);
    std::cout << "[sentinel] status = " << to_string(status)
              << ", iter = " << solver.getSolverStatistics().iter << std::endl;
  }

  // (b) masked bounds (disable the inactive sides via mask = 0)
  {
    auto qp = build();
    for (int i=0; i<N; ++i) {
      qp[i].idxbu = {0, 1, 2};
      qp[i].lbu = VectorXf::Zero(nu);
      qp[i].ubu = VectorXf::Zero(nu);
      qp[i].lbu_mask = VectorXf::Zero(nu);  // 0 => bound disabled
      qp[i].ubu_mask = VectorXf::Zero(nu);
    }
    std::vector<OcpQpSolutionTpl<float>> sol(N+1);
    OcpQpIpmSolverTpl<float> solver(qp, settings);
    const auto status = solver.solve(x0, qp, sol);
    std::cout << "[masked]   status = " << to_string(status)
              << ", iter = " << solver.getSolverStatistics().iter << std::endl;
  }
}

} // namespace hpipm


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

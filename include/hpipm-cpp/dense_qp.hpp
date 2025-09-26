#ifndef HPIPM_CPP_DENSE_QP_HPP_
#define HPIPM_CPP_DENSE_QP_HPP_

extern "C"{
#include "hpipm_d_dense_qp_ipm.h"
}

#include <numeric>
#include "Eigen/Core"
#include <memory>
#include <iostream>

namespace hpipm
{
  class DenseQpSolver
  {
  public:
    DenseQpSolver(int dim_var,
                  int dim_eq,
                  int dim_ineq);
    ~DenseQpSolver() = default;

    std::unique_ptr<struct d_dense_qp_dim> qp_dim_;
    std::unique_ptr<struct d_dense_qp> qp_;
    std::unique_ptr<struct d_dense_qp_sol> qp_sol_;
    std::unique_ptr<struct d_dense_qp_ipm_arg> ipm_arg_;
    std::unique_ptr<struct d_dense_qp_ipm_ws> ipm_ws_;

    std::unique_ptr<uint8_t[]> qp_dim_mem_ = nullptr;
    std::unique_ptr<uint8_t[]> qp_mem_ = nullptr;
    std::unique_ptr<uint8_t[]> qp_sol_mem_ = nullptr;
    std::unique_ptr<uint8_t[]> ipm_arg_mem_ = nullptr;
    std::unique_ptr<uint8_t[]> ipm_ws_mem_ = nullptr;
    std::unique_ptr<double[]> opt_x_mem_ = nullptr;
    std::unique_ptr<double[]> opt_lam_lb_mem_ = nullptr;
    std::unique_ptr<double[]> opt_lam_ub_mem_ = nullptr;
    std::unique_ptr<double[]> opt_lam_lg_mem_ = nullptr;
    std::unique_ptr<double[]> opt_lam_ug_mem_ = nullptr;

    ///
    /// @brief Hessian for cost
    ///
    Eigen::MatrixXd H_;

    ///
    /// @brief Gradient for cost
    ///
    Eigen::VectorXd g_;

    ///
    /// @brief Equality constraint matrix
    ///
    Eigen::MatrixXd A_;

    ///
    /// @brief Equality constraint vector
    ///
    Eigen::VectorXd b_;

    ///
    /// @brief Inequality constraint matrix
    ///
    Eigen::MatrixXd C_;

    ///
    /// @brief Ineuqality constraint lower bounds vector
    ///
    Eigen::VectorXd lbg_;

    ///
    /// @brief Ineuqality constraint upper bounds vector
    ///
    Eigen::VectorXd ubg_;

    ///
    /// @brief Lower bounds of variables
    ///
    Eigen::VectorXd lbx_;

    ///
    /// @brief Upper bounds of variables
    ///
    Eigen::VectorXd ubx_;

    ///
    /// @brief QP solution
    ///
    Eigen::VectorXd opt_x_;

    ///
    /// @brief Dual solution
    ///
    Eigen::VectorXd opt_lam_lb_;
    Eigen::VectorXd opt_lam_ub_;
    Eigen::VectorXd opt_lam_b_;
    Eigen::VectorXd opt_lam_lg_;
    Eigen::VectorXd opt_lam_ug_;
    Eigen::VectorXd opt_lam_g_;

    void solve(Eigen::Ref<Eigen::MatrixXd> H,
               const Eigen::Ref<const Eigen::VectorXd> & g,
               const Eigen::Ref<const Eigen::MatrixXd> & A,
               const Eigen::Ref<const Eigen::VectorXd> & b,
               const Eigen::Ref<const Eigen::MatrixXd> & C,
               const Eigen::Ref<const Eigen::VectorXd> & lbg,
               const Eigen::Ref<const Eigen::VectorXd> & ubg,
               const Eigen::Ref<const Eigen::VectorXd> & lbx,
               const Eigen::Ref<const Eigen::VectorXd> & ubx);
    void solve();
    Eigen::VectorXd getOptX() { return opt_x_; }
    Eigen::VectorXd getOptLamB() { return opt_lam_b_; }
    Eigen::VectorXd getOptLamG() { return opt_lam_g_; }

  private:
    int dim_var_;
    int dim_eq_;
    int dim_ineq_;
    double bound_limit_ = 1e10;
  };
}

#endif // HPIPM_CPP_DENSE_QP_HPP_

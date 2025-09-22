#include "hpipm-cpp/dense_qp.hpp"

namespace hpipm
{
  DenseQpSolver::DenseQpSolver(int dim_var,
                               int dim_eq,
                               int dim_ineq)
  {
    dim_var_ = dim_var;
    dim_eq_ = dim_eq;
    dim_ineq_ = dim_ineq;

    qp_dim_ = std::make_unique<struct d_dense_qp_dim>();
    qp_ = std::make_unique<struct d_dense_qp>();
    qp_sol_ = std::make_unique<struct d_dense_qp_sol>();
    ipm_arg_ = std::make_unique<struct d_dense_qp_ipm_arg>();
    ipm_ws_ = std::make_unique<struct d_dense_qp_ipm_ws>();

    int qp_dim_size = d_dense_qp_dim_memsize();
    qp_dim_mem_ = std::make_unique<uint8_t[]>(qp_dim_size);
    d_dense_qp_dim_create(qp_dim_.get(), qp_dim_mem_.get());
    d_dense_qp_dim_set_all(dim_var, dim_eq, dim_var, dim_ineq, 0, qp_dim_.get());

    int qp_size = d_dense_qp_memsize(qp_dim_.get());
    qp_mem_ = std::make_unique<uint8_t[]>(qp_size);
    d_dense_qp_create(qp_dim_.get(), qp_.get(), qp_mem_.get());

    int qp_sol_size = d_dense_qp_sol_memsize(qp_dim_.get());
    qp_sol_mem_ = std::make_unique<uint8_t[]>(qp_sol_size);
    d_dense_qp_sol_create(qp_dim_.get(), qp_sol_.get(), qp_sol_mem_.get());

    int ipm_arg_size = d_dense_qp_ipm_arg_memsize(qp_dim_.get());
    ipm_arg_mem_ = std::make_unique<uint8_t[]>(ipm_arg_size);
    d_dense_qp_ipm_arg_create(qp_dim_.get(), ipm_arg_.get(), ipm_arg_mem_.get());
    enum hpipm_mode mode = BALANCE; // SPEED_ABS, SPEED, BALANCE, ROBUST
    d_dense_qp_ipm_arg_set_default(mode, ipm_arg_.get());

    int ipm_ws_size = d_dense_qp_ipm_ws_memsize(qp_dim_.get(), ipm_arg_.get());
    ipm_ws_mem_ = std::make_unique<uint8_t[]>(ipm_ws_size);
    d_dense_qp_ipm_ws_create(qp_dim_.get(), ipm_arg_.get(), ipm_ws_.get(), ipm_ws_mem_.get());

    opt_x_mem_ = std::make_unique<double[]>(dim_var); // Automatic memory management for the array
  }

  Eigen::VectorXd DenseQpSolver::solve(Eigen::Ref<Eigen::MatrixXd> H,
                                       const Eigen::Ref<const Eigen::VectorXd> & g,
                                       const Eigen::Ref<const Eigen::MatrixXd> & A,
                                       const Eigen::Ref<const Eigen::VectorXd> & b,
                                       const Eigen::Ref<const Eigen::MatrixXd> & C,
                                       const Eigen::Ref<const Eigen::VectorXd> & lbg,
                                       const Eigen::Ref<const Eigen::VectorXd> & ubg,
                                       const Eigen::Ref<const Eigen::VectorXd> & lbx,
                                       const Eigen::Ref<const Eigen::VectorXd> & ubx)
  {
    // set QP coefficients
    d_dense_qp_set_H(H.data(), qp_.get());
    d_dense_qp_set_g(const_cast<double *>(g.data()), qp_.get());
    d_dense_qp_set_A(const_cast<double *>(A.data()), qp_.get());
    d_dense_qp_set_b(const_cast<double *>(b.data()), qp_.get());
    d_dense_qp_set_C(const_cast<double *>(C.data()), qp_.get());
    d_dense_qp_set_lg(const_cast<double *>(lbg.cwiseMax(-bound_limit_).eval().data()), qp_.get());
    d_dense_qp_set_ug(const_cast<double *>(ubg.cwiseMin(bound_limit_).eval().data()), qp_.get());
    std::vector<int> idxb(dim_var_);
    std::iota(idxb.begin(), idxb.end(), 0);
    d_dense_qp_set_idxb(idxb.data(), qp_.get());
    d_dense_qp_set_lb(const_cast<double *>(lbx.cwiseMax(-1 * bound_limit_).eval().data()), qp_.get());
    d_dense_qp_set_ub(const_cast<double *>(ubx.cwiseMin(bound_limit_).eval().data()), qp_.get());

    // Solve QP
    d_dense_qp_ipm_solve(qp_.get(), qp_sol_.get(), ipm_arg_.get(), ipm_ws_.get());
    d_dense_qp_sol_get_v(qp_sol_.get(), opt_x_mem_.get());

    int status;
    d_dense_qp_ipm_get_status(ipm_ws_.get(), &status);
    if(status == SUCCESS || status == MAX_ITER) // enum hpipm_status
    {}
    else
    {
      std::cerr << "[QpSolverHpipm::solve] Failed to solve: " << status << std::endl;
    }

    return Eigen::Map<Eigen::VectorXd>(opt_x_mem_.get(), dim_var_);
  }
}

#ifndef HPIPM_CPP_DETAIL_HPIPM_TRAITS_HPP_
#define HPIPM_CPP_DETAIL_HPIPM_TRAITS_HPP_

extern "C" {
#include "hpipm_d_ocp_qp_dim.h"
#include "hpipm_d_ocp_qp.h"
#include "hpipm_d_ocp_qp_sol.h"
#include "hpipm_d_ocp_qp_ipm.h"

#include "hpipm_s_ocp_qp_dim.h"
#include "hpipm_s_ocp_qp.h"
#include "hpipm_s_ocp_qp_sol.h"
#include "hpipm_s_ocp_qp_ipm.h"
}


namespace hpipm {

///
/// @struct hpipm_traits
/// @brief Maps a scalar type (double / float) to the corresponding hpipm
/// struct types and functions. Every detail wrapper and the IPM solver are
/// written once in terms of this trait, so the precision-dependent `d_` / `s_`
/// hpipm symbols are dispatched here instead of being hard-coded.
///
/// Scalar-valued arguments are declared as `Scalar*` so the templated callers
/// can forward `Eigen::Matrix<Scalar>::data()` directly; the precision always
/// matches the selected hpipm symbol.
///
template <typename Scalar>
struct hpipm_traits;


template <>
struct hpipm_traits<double> {
  using ocp_qp_dim     = d_ocp_qp_dim;
  using ocp_qp         = d_ocp_qp;
  using ocp_qp_sol     = d_ocp_qp_sol;
  using ocp_qp_ipm_arg = d_ocp_qp_ipm_arg;
  using ocp_qp_ipm_ws  = d_ocp_qp_ipm_ws;

  // --- dim ---
  static hpipm_size_t ocp_qp_dim_memsize(const int N) {
    return d_ocp_qp_dim_memsize(N);
  }
  static void ocp_qp_dim_create(const int N, ocp_qp_dim* dim, void* memory) {
    d_ocp_qp_dim_create(N, dim, memory);
  }
  static void ocp_qp_dim_copy_all(ocp_qp_dim* orig, ocp_qp_dim* dest) {
    d_ocp_qp_dim_copy_all(orig, dest);
  }
  static void ocp_qp_dim_set_all(int* nx, int* nu, int* nbx, int* nbu, int* ng,
                                 int* ns, ocp_qp_dim* dim) {
    d_ocp_qp_dim_set_all(nx, nu, nbx, nbu, ng, ns, dim);
  }
  static void ocp_qp_dim_set_nx(int stage, int value, ocp_qp_dim* dim) {
    d_ocp_qp_dim_set_nx(stage, value, dim);
  }
  static void ocp_qp_dim_set_nbx(int stage, int value, ocp_qp_dim* dim) {
    d_ocp_qp_dim_set_nbx(stage, value, dim);
  }
  static void ocp_qp_dim_set_ns(int stage, int value, ocp_qp_dim* dim) {
    d_ocp_qp_dim_set_ns(stage, value, dim);
  }

  // --- qp ---
  static hpipm_size_t ocp_qp_memsize(ocp_qp_dim* dim) {
    return d_ocp_qp_memsize(dim);
  }
  static void ocp_qp_create(ocp_qp_dim* dim, ocp_qp* qp, void* memory) {
    d_ocp_qp_create(dim, qp, memory);
  }
  static void ocp_qp_copy_all(ocp_qp* orig, ocp_qp* dest) {
    d_ocp_qp_copy_all(orig, dest);
  }
  static void ocp_qp_set_all(double** A, double** B, double** b, double** Q,
                             double** S, double** R, double** q, double** r,
                             int** idxbx, double** lbx, double** ubx,
                             int** idxbu, double** lbu, double** ubu,
                             double** C, double** D, double** lg, double** ug,
                             double** Zl, double** Zu, double** zl, double** zu,
                             int** idxs, int** idxs_rev, double** ls,
                             double** us, ocp_qp* qp) {
    d_ocp_qp_set_all(A, B, b, Q, S, R, q, r, idxbx, lbx, ubx, idxbu, lbu, ubu,
                     C, D, lg, ug, Zl, Zu, zl, zu, idxs, idxs_rev, ls, us, qp);
  }
  static void ocp_qp_set_lbx_mask(int stage, double* vec, ocp_qp* qp) {
    d_ocp_qp_set_lbx_mask(stage, vec, qp);
  }
  static void ocp_qp_set_ubx_mask(int stage, double* vec, ocp_qp* qp) {
    d_ocp_qp_set_ubx_mask(stage, vec, qp);
  }
  static void ocp_qp_set_lbu_mask(int stage, double* vec, ocp_qp* qp) {
    d_ocp_qp_set_lbu_mask(stage, vec, qp);
  }
  static void ocp_qp_set_ubu_mask(int stage, double* vec, ocp_qp* qp) {
    d_ocp_qp_set_ubu_mask(stage, vec, qp);
  }
  static void ocp_qp_set_lg_mask(int stage, double* vec, ocp_qp* qp) {
    d_ocp_qp_set_lg_mask(stage, vec, qp);
  }
  static void ocp_qp_set_ug_mask(int stage, double* vec, ocp_qp* qp) {
    d_ocp_qp_set_ug_mask(stage, vec, qp);
  }

  // --- sol ---
  static hpipm_size_t ocp_qp_sol_memsize(ocp_qp_dim* dim) {
    return d_ocp_qp_sol_memsize(dim);
  }
  static void ocp_qp_sol_create(ocp_qp_dim* dim, ocp_qp_sol* sol, void* memory) {
    d_ocp_qp_sol_create(dim, sol, memory);
  }
  static void ocp_qp_sol_copy_all(ocp_qp_sol* orig, ocp_qp_sol* dest) {
    d_ocp_qp_sol_copy_all(orig, dest);
  }
  static void ocp_qp_sol_set_x(int stage, double* vec, ocp_qp_sol* sol) {
    d_ocp_qp_sol_set_x(stage, vec, sol);
  }
  static void ocp_qp_sol_set_u(int stage, double* vec, ocp_qp_sol* sol) {
    d_ocp_qp_sol_set_u(stage, vec, sol);
  }
  static void ocp_qp_sol_get_x(int stage, ocp_qp_sol* sol, double* vec) {
    d_ocp_qp_sol_get_x(stage, sol, vec);
  }
  static void ocp_qp_sol_get_u(int stage, ocp_qp_sol* sol, double* vec) {
    d_ocp_qp_sol_get_u(stage, sol, vec);
  }
  static void ocp_qp_sol_get_pi(int stage, ocp_qp_sol* sol, double* vec) {
    d_ocp_qp_sol_get_pi(stage, sol, vec);
  }

  // --- ipm arg ---
  static hpipm_size_t ocp_qp_ipm_arg_memsize(ocp_qp_dim* dim) {
    return d_ocp_qp_ipm_arg_memsize(dim);
  }
  static void ocp_qp_ipm_arg_create(ocp_qp_dim* dim, ocp_qp_ipm_arg* arg, void* memory) {
    d_ocp_qp_ipm_arg_create(dim, arg, memory);
  }
  static void ocp_qp_ipm_arg_set_default(enum hpipm_mode mode, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_default(mode, arg);
  }
  static void ocp_qp_ipm_arg_set_mu0(double* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_mu0(v, arg);
  }
  static void ocp_qp_ipm_arg_set_iter_max(int* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_iter_max(v, arg);
  }
  static void ocp_qp_ipm_arg_set_alpha_min(double* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_alpha_min(v, arg);
  }
  static void ocp_qp_ipm_arg_set_tol_stat(double* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_tol_stat(v, arg);
  }
  static void ocp_qp_ipm_arg_set_tol_eq(double* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_tol_eq(v, arg);
  }
  static void ocp_qp_ipm_arg_set_tol_ineq(double* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_tol_ineq(v, arg);
  }
  static void ocp_qp_ipm_arg_set_tol_comp(double* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_tol_comp(v, arg);
  }
  static void ocp_qp_ipm_arg_set_reg_prim(double* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_reg_prim(v, arg);
  }
  static void ocp_qp_ipm_arg_set_warm_start(int* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_warm_start(v, arg);
  }
  static void ocp_qp_ipm_arg_set_pred_corr(int* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_pred_corr(v, arg);
  }
  static void ocp_qp_ipm_arg_set_ric_alg(int* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_ric_alg(v, arg);
  }
  static void ocp_qp_ipm_arg_set_split_step(int* v, ocp_qp_ipm_arg* arg) {
    d_ocp_qp_ipm_arg_set_split_step(v, arg);
  }

  // --- ipm ws ---
  static hpipm_size_t ocp_qp_ipm_ws_memsize(ocp_qp_dim* dim, ocp_qp_ipm_arg* arg) {
    return d_ocp_qp_ipm_ws_memsize(dim, arg);
  }
  static void ocp_qp_ipm_ws_create(ocp_qp_dim* dim, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, void* memory) {
    d_ocp_qp_ipm_ws_create(dim, arg, ws, memory);
  }

  // --- ipm solve / get ---
  static void ocp_qp_ipm_solve(ocp_qp* qp, ocp_qp_sol* sol, ocp_qp_ipm_arg* arg,
                               ocp_qp_ipm_ws* ws) {
    d_ocp_qp_ipm_solve(qp, sol, arg, ws);
  }
  static void ocp_qp_ipm_get_status(ocp_qp_ipm_ws* ws, int* status) {
    d_ocp_qp_ipm_get_status(ws, status);
  }
  static void ocp_qp_ipm_get_iter(ocp_qp_ipm_ws* ws, int* iter) {
    d_ocp_qp_ipm_get_iter(ws, iter);
  }
  static void ocp_qp_ipm_get_max_res_stat(ocp_qp_ipm_ws* ws, double* v) {
    d_ocp_qp_ipm_get_max_res_stat(ws, v);
  }
  static void ocp_qp_ipm_get_max_res_eq(ocp_qp_ipm_ws* ws, double* v) {
    d_ocp_qp_ipm_get_max_res_eq(ws, v);
  }
  static void ocp_qp_ipm_get_max_res_ineq(ocp_qp_ipm_ws* ws, double* v) {
    d_ocp_qp_ipm_get_max_res_ineq(ws, v);
  }
  static void ocp_qp_ipm_get_max_res_comp(ocp_qp_ipm_ws* ws, double* v) {
    d_ocp_qp_ipm_get_max_res_comp(ws, v);
  }
  static void ocp_qp_ipm_get_ric_Lr(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                    ocp_qp_ipm_ws* ws, int stage, double* v) {
    d_ocp_qp_ipm_get_ric_Lr(qp, arg, ws, stage, v);
  }
  static void ocp_qp_ipm_get_ric_P(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, int stage, double* v) {
    d_ocp_qp_ipm_get_ric_P(qp, arg, ws, stage, v);
  }
  static void ocp_qp_ipm_get_ric_p(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, int stage, double* v) {
    d_ocp_qp_ipm_get_ric_p(qp, arg, ws, stage, v);
  }
  static void ocp_qp_ipm_get_ric_K(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, int stage, double* v) {
    d_ocp_qp_ipm_get_ric_K(qp, arg, ws, stage, v);
  }
  static void ocp_qp_ipm_get_ric_k(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, int stage, double* v) {
    d_ocp_qp_ipm_get_ric_k(qp, arg, ws, stage, v);
  }
};


template <>
struct hpipm_traits<float> {
  using ocp_qp_dim     = s_ocp_qp_dim;
  using ocp_qp         = s_ocp_qp;
  using ocp_qp_sol     = s_ocp_qp_sol;
  using ocp_qp_ipm_arg = s_ocp_qp_ipm_arg;
  using ocp_qp_ipm_ws  = s_ocp_qp_ipm_ws;

  // --- dim ---
  static hpipm_size_t ocp_qp_dim_memsize(const int N) {
    return s_ocp_qp_dim_memsize(N);
  }
  static void ocp_qp_dim_create(const int N, ocp_qp_dim* dim, void* memory) {
    s_ocp_qp_dim_create(N, dim, memory);
  }
  static void ocp_qp_dim_copy_all(ocp_qp_dim* orig, ocp_qp_dim* dest) {
    s_ocp_qp_dim_copy_all(orig, dest);
  }
  static void ocp_qp_dim_set_all(int* nx, int* nu, int* nbx, int* nbu, int* ng,
                                 int* ns, ocp_qp_dim* dim) {
    s_ocp_qp_dim_set_all(nx, nu, nbx, nbu, ng, ns, dim);
  }
  static void ocp_qp_dim_set_nx(int stage, int value, ocp_qp_dim* dim) {
    s_ocp_qp_dim_set_nx(stage, value, dim);
  }
  static void ocp_qp_dim_set_nbx(int stage, int value, ocp_qp_dim* dim) {
    s_ocp_qp_dim_set_nbx(stage, value, dim);
  }
  static void ocp_qp_dim_set_ns(int stage, int value, ocp_qp_dim* dim) {
    s_ocp_qp_dim_set_ns(stage, value, dim);
  }

  // --- qp ---
  static hpipm_size_t ocp_qp_memsize(ocp_qp_dim* dim) {
    return s_ocp_qp_memsize(dim);
  }
  static void ocp_qp_create(ocp_qp_dim* dim, ocp_qp* qp, void* memory) {
    s_ocp_qp_create(dim, qp, memory);
  }
  static void ocp_qp_copy_all(ocp_qp* orig, ocp_qp* dest) {
    s_ocp_qp_copy_all(orig, dest);
  }
  static void ocp_qp_set_all(float** A, float** B, float** b, float** Q,
                             float** S, float** R, float** q, float** r,
                             int** idxbx, float** lbx, float** ubx,
                             int** idxbu, float** lbu, float** ubu,
                             float** C, float** D, float** lg, float** ug,
                             float** Zl, float** Zu, float** zl, float** zu,
                             int** idxs, int** idxs_rev, float** ls,
                             float** us, ocp_qp* qp) {
    s_ocp_qp_set_all(A, B, b, Q, S, R, q, r, idxbx, lbx, ubx, idxbu, lbu, ubu,
                     C, D, lg, ug, Zl, Zu, zl, zu, idxs, idxs_rev, ls, us, qp);
  }
  static void ocp_qp_set_lbx_mask(int stage, float* vec, ocp_qp* qp) {
    s_ocp_qp_set_lbx_mask(stage, vec, qp);
  }
  static void ocp_qp_set_ubx_mask(int stage, float* vec, ocp_qp* qp) {
    s_ocp_qp_set_ubx_mask(stage, vec, qp);
  }
  static void ocp_qp_set_lbu_mask(int stage, float* vec, ocp_qp* qp) {
    s_ocp_qp_set_lbu_mask(stage, vec, qp);
  }
  static void ocp_qp_set_ubu_mask(int stage, float* vec, ocp_qp* qp) {
    s_ocp_qp_set_ubu_mask(stage, vec, qp);
  }
  static void ocp_qp_set_lg_mask(int stage, float* vec, ocp_qp* qp) {
    s_ocp_qp_set_lg_mask(stage, vec, qp);
  }
  static void ocp_qp_set_ug_mask(int stage, float* vec, ocp_qp* qp) {
    s_ocp_qp_set_ug_mask(stage, vec, qp);
  }

  // --- sol ---
  static hpipm_size_t ocp_qp_sol_memsize(ocp_qp_dim* dim) {
    return s_ocp_qp_sol_memsize(dim);
  }
  static void ocp_qp_sol_create(ocp_qp_dim* dim, ocp_qp_sol* sol, void* memory) {
    s_ocp_qp_sol_create(dim, sol, memory);
  }
  static void ocp_qp_sol_copy_all(ocp_qp_sol* orig, ocp_qp_sol* dest) {
    s_ocp_qp_sol_copy_all(orig, dest);
  }
  static void ocp_qp_sol_set_x(int stage, float* vec, ocp_qp_sol* sol) {
    s_ocp_qp_sol_set_x(stage, vec, sol);
  }
  static void ocp_qp_sol_set_u(int stage, float* vec, ocp_qp_sol* sol) {
    s_ocp_qp_sol_set_u(stage, vec, sol);
  }
  static void ocp_qp_sol_get_x(int stage, ocp_qp_sol* sol, float* vec) {
    s_ocp_qp_sol_get_x(stage, sol, vec);
  }
  static void ocp_qp_sol_get_u(int stage, ocp_qp_sol* sol, float* vec) {
    s_ocp_qp_sol_get_u(stage, sol, vec);
  }
  static void ocp_qp_sol_get_pi(int stage, ocp_qp_sol* sol, float* vec) {
    s_ocp_qp_sol_get_pi(stage, sol, vec);
  }

  // --- ipm arg ---
  static hpipm_size_t ocp_qp_ipm_arg_memsize(ocp_qp_dim* dim) {
    return s_ocp_qp_ipm_arg_memsize(dim);
  }
  static void ocp_qp_ipm_arg_create(ocp_qp_dim* dim, ocp_qp_ipm_arg* arg, void* memory) {
    s_ocp_qp_ipm_arg_create(dim, arg, memory);
  }
  static void ocp_qp_ipm_arg_set_default(enum hpipm_mode mode, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_default(mode, arg);
  }
  static void ocp_qp_ipm_arg_set_mu0(float* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_mu0(v, arg);
  }
  static void ocp_qp_ipm_arg_set_iter_max(int* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_iter_max(v, arg);
  }
  static void ocp_qp_ipm_arg_set_alpha_min(float* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_alpha_min(v, arg);
  }
  static void ocp_qp_ipm_arg_set_tol_stat(float* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_tol_stat(v, arg);
  }
  static void ocp_qp_ipm_arg_set_tol_eq(float* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_tol_eq(v, arg);
  }
  static void ocp_qp_ipm_arg_set_tol_ineq(float* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_tol_ineq(v, arg);
  }
  static void ocp_qp_ipm_arg_set_tol_comp(float* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_tol_comp(v, arg);
  }
  static void ocp_qp_ipm_arg_set_reg_prim(float* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_reg_prim(v, arg);
  }
  static void ocp_qp_ipm_arg_set_warm_start(int* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_warm_start(v, arg);
  }
  static void ocp_qp_ipm_arg_set_pred_corr(int* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_pred_corr(v, arg);
  }
  static void ocp_qp_ipm_arg_set_ric_alg(int* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_ric_alg(v, arg);
  }
  static void ocp_qp_ipm_arg_set_split_step(int* v, ocp_qp_ipm_arg* arg) {
    s_ocp_qp_ipm_arg_set_split_step(v, arg);
  }

  // --- ipm ws ---
  static hpipm_size_t ocp_qp_ipm_ws_memsize(ocp_qp_dim* dim, ocp_qp_ipm_arg* arg) {
    return s_ocp_qp_ipm_ws_memsize(dim, arg);
  }
  static void ocp_qp_ipm_ws_create(ocp_qp_dim* dim, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, void* memory) {
    s_ocp_qp_ipm_ws_create(dim, arg, ws, memory);
  }

  // --- ipm solve / get ---
  static void ocp_qp_ipm_solve(ocp_qp* qp, ocp_qp_sol* sol, ocp_qp_ipm_arg* arg,
                               ocp_qp_ipm_ws* ws) {
    s_ocp_qp_ipm_solve(qp, sol, arg, ws);
  }
  static void ocp_qp_ipm_get_status(ocp_qp_ipm_ws* ws, int* status) {
    s_ocp_qp_ipm_get_status(ws, status);
  }
  static void ocp_qp_ipm_get_iter(ocp_qp_ipm_ws* ws, int* iter) {
    s_ocp_qp_ipm_get_iter(ws, iter);
  }
  static void ocp_qp_ipm_get_max_res_stat(ocp_qp_ipm_ws* ws, float* v) {
    s_ocp_qp_ipm_get_max_res_stat(ws, v);
  }
  static void ocp_qp_ipm_get_max_res_eq(ocp_qp_ipm_ws* ws, float* v) {
    s_ocp_qp_ipm_get_max_res_eq(ws, v);
  }
  static void ocp_qp_ipm_get_max_res_ineq(ocp_qp_ipm_ws* ws, float* v) {
    s_ocp_qp_ipm_get_max_res_ineq(ws, v);
  }
  static void ocp_qp_ipm_get_max_res_comp(ocp_qp_ipm_ws* ws, float* v) {
    s_ocp_qp_ipm_get_max_res_comp(ws, v);
  }
  static void ocp_qp_ipm_get_ric_Lr(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                    ocp_qp_ipm_ws* ws, int stage, float* v) {
    s_ocp_qp_ipm_get_ric_Lr(qp, arg, ws, stage, v);
  }
  static void ocp_qp_ipm_get_ric_P(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, int stage, float* v) {
    s_ocp_qp_ipm_get_ric_P(qp, arg, ws, stage, v);
  }
  static void ocp_qp_ipm_get_ric_p(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, int stage, float* v) {
    s_ocp_qp_ipm_get_ric_p(qp, arg, ws, stage, v);
  }
  static void ocp_qp_ipm_get_ric_K(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, int stage, float* v) {
    s_ocp_qp_ipm_get_ric_K(qp, arg, ws, stage, v);
  }
  static void ocp_qp_ipm_get_ric_k(ocp_qp* qp, ocp_qp_ipm_arg* arg,
                                   ocp_qp_ipm_ws* ws, int stage, float* v) {
    s_ocp_qp_ipm_get_ric_k(qp, arg, ws, stage, v);
  }
};

} // namespace hpipm

#endif // HPIPM_CPP_DETAIL_HPIPM_TRAITS_HPP_

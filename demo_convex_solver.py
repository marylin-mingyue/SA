import numpy as np

from mini_convex_solver import SolverOptions, solve_lp, solve_qp, solve_qp_sparse

try:
    import scipy.sparse as sp

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def run_lp_demo() -> None:
    print("=== LP demo ===")
    # minimize c^T x
    # s.t. x1 + x2 + x3 = 1, x >= 0
    c = np.array([1.0, 2.0, 3.0])
    A = np.array([[1.0, 1.0, 1.0]])
    b = np.array([1.0])

    result = solve_lp(
        c,
        A,
        b,
        options=SolverOptions(max_iters=60, tol=1e-9, alpha_safety=0.995, regularization=1e-10),
    )
    print(f"converged          : {result.converged}")
    print(f"iterations         : {result.iterations}")
    print(f"objective          : {result.objective:.10f}")
    print(f"x                  : {result.x}")
    print(f"primal residual    : {result.primal_residual:.3e}")
    print(f"dual residual      : {result.dual_residual:.3e}")
    print(f"complementarity(mu): {result.complementarity:.3e}")
    print()


def run_qp_demo() -> None:
    print("=== QP demo ===")
    # minimize 0.5 x^T Q x + c^T x
    # s.t. x1 + x2 = 1, x >= 0
    Q = np.array([[4.0, 1.0], [1.0, 2.0]])
    c = np.array([-8.0, -3.0])
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])

    result = solve_qp(
        Q,
        c,
        A,
        b,
        options=SolverOptions(max_iters=80, tol=1e-9, alpha_safety=0.995, regularization=1e-10),
    )
    print(f"converged          : {result.converged}")
    print(f"iterations         : {result.iterations}")
    print(f"objective          : {result.objective:.10f}")
    print(f"x                  : {result.x}")
    print(f"primal residual    : {result.primal_residual:.3e}")
    print(f"dual residual      : {result.dual_residual:.3e}")
    print(f"complementarity(mu): {result.complementarity:.3e}")
    print()


def run_sparse_qp_demo() -> None:
    if not SCIPY_AVAILABLE:
        print("=== Sparse QP demo ===")
        print("scipy not installed, skipped sparse demo.")
        print()
        return

    print("=== Sparse QP demo (engineering-scale entry) ===")
    rng = np.random.default_rng(2026)
    n = 600
    m = 120
    density = 0.02

    x_star = rng.uniform(0.2, 1.2, size=n)
    A = sp.random(
        m,
        n,
        density=density,
        format="csc",
        random_state=2026,
        data_rvs=lambda k: rng.uniform(0.2, 1.0, size=k),
    )
    b = A @ x_star

    q_diag = rng.uniform(0.5, 2.0, size=n)
    Q = sp.diags(q_diag, offsets=0, format="csc")
    y_star = rng.normal(0.0, 0.1, size=m)
    c = -(q_diag * x_star) + np.asarray(A.T @ y_star).reshape(-1)

    result = solve_qp_sparse(
        Q,
        c,
        A,
        b,
        options=SolverOptions(max_iters=80, tol=1e-8, alpha_safety=0.995, regularization=1e-10),
        h_solver="splu",
        schur_solver="cg",
        cg_tol=1e-9,
        cg_max_iters=800,
    )
    print(f"converged          : {result.converged}")
    print(f"iterations         : {result.iterations}")
    print(f"objective          : {result.objective:.10f}")
    print(f"||x - x_star||_2   : {np.linalg.norm(result.x - x_star):.3e}")
    print(f"primal residual    : {result.primal_residual:.3e}")
    print(f"dual residual      : {result.dual_residual:.3e}")
    print(f"complementarity(mu): {result.complementarity:.3e}")
    print()


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    run_lp_demo()
    run_qp_demo()
    run_sparse_qp_demo()


if __name__ == "__main__":
    main()

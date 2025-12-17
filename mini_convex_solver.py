from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    SCIPY_AVAILABLE = True
except Exception:
    sp = None
    spla = None
    SCIPY_AVAILABLE = False


@dataclass
class QPProblem:
    """Quadratic program in standard primal form.

    minimize    0.5 * x^T Q x + c^T x
    subject to  A x = b
                x >= 0
    """

    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray


@dataclass
class SolverOptions:
    max_iters: int = 80
    tol: float = 1e-8
    alpha_safety: float = 0.99
    verbose: bool = False
    regularization: float = 1e-9


@dataclass
class SolverResult:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    objective: float
    converged: bool
    iterations: int
    primal_residual: float
    dual_residual: float
    complementarity: float


def _as_float_array(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float)


def _as_float_vector(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float).reshape(-1)


def _require_scipy() -> None:
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "Sparse solver requires scipy. Install it with `python3 -m pip install scipy`."
        )


def _cholesky_decompose_spd(matrix: np.ndarray) -> np.ndarray:
    """Dense Cholesky decomposition (A = L L^T) for SPD matrix."""
    a = _as_float_array(matrix)
    n = a.shape[0]
    l = np.zeros_like(a)

    for i in range(n):
        for j in range(i + 1):
            s = a[i, j]
            for k in range(j):
                s -= l[i, k] * l[j, k]

            if i == j:
                if s <= 0.0:
                    raise np.linalg.LinAlgError("Matrix is not SPD in custom Cholesky.")
                l[i, j] = np.sqrt(s)
            else:
                l[i, j] = s / l[j, j]
    return l


def _cholesky_solve(l: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    rhs_2d = rhs if rhs.ndim == 2 else rhs[:, None]
    n, nrhs = rhs_2d.shape

    # Forward solve: L y = rhs
    y = np.zeros_like(rhs_2d)
    for col in range(nrhs):
        for i in range(n):
            s = rhs_2d[i, col]
            for k in range(i):
                s -= l[i, k] * y[k, col]
            y[i, col] = s / l[i, i]

    # Backward solve: L^T x = y
    x = np.zeros_like(rhs_2d)
    for col in range(nrhs):
        for i in range(n - 1, -1, -1):
            s = y[i, col]
            for k in range(i + 1, n):
                s -= l[k, i] * x[k, col]
            x[i, col] = s / l[i, i]

    return x if rhs.ndim == 2 else x[:, 0]


def _solve_spd(matrix: np.ndarray, rhs: np.ndarray, regularization: float = 0.0) -> np.ndarray:
    mat = _as_float_array(matrix).copy()
    if regularization > 0.0:
        mat += regularization * np.eye(mat.shape[0])

    try:
        l = _cholesky_decompose_spd(mat)
        return _cholesky_solve(l, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.solve(mat, rhs)


def _max_step_to_positive(current: np.ndarray, direction: np.ndarray, alpha_safety: float) -> float:
    mask = direction < 0.0
    if not np.any(mask):
        return 1.0
    alpha = np.min(-current[mask] / direction[mask])
    return min(1.0, alpha_safety * alpha)


def _build_sparse_linear_solver(
    matrix,
    method: str,
    cg_tol: float,
    cg_max_iters: int,
) -> Callable[[np.ndarray], np.ndarray]:
    if method not in ("splu", "cg"):
        raise ValueError("method must be one of: 'splu', 'cg'.")

    if method == "splu":
        lu = spla.splu(matrix.tocsc())

        def solve(rhs: np.ndarray) -> np.ndarray:
            rhs_arr = np.asarray(rhs, dtype=float)
            if rhs_arr.ndim == 1:
                return lu.solve(rhs_arr)
            out = np.zeros_like(rhs_arr, dtype=float)
            for col in range(rhs_arr.shape[1]):
                out[:, col] = lu.solve(rhs_arr[:, col])
            return out

        return solve

    def solve(rhs: np.ndarray) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float)
        if rhs_arr.ndim == 1:
            sol, info = _cg_solve_compat(matrix, rhs_arr, cg_tol, cg_max_iters)
            if info != 0:
                raise RuntimeError(f"CG failed for H-system (info={info}).")
            return sol
        out = np.zeros_like(rhs_arr, dtype=float)
        for col in range(rhs_arr.shape[1]):
            sol, info = _cg_solve_compat(matrix, rhs_arr[:, col], cg_tol, cg_max_iters)
            if info != 0:
                raise RuntimeError(f"CG failed for H-system column {col} (info={info}).")
            out[:, col] = sol
        return out

    return solve


def _cg_solve_compat(linear_operator, rhs: np.ndarray, tol: float, maxiter: int):
    try:
        return spla.cg(linear_operator, rhs, tol=tol, maxiter=maxiter)
    except TypeError:
        # scipy>=1.14 uses rtol/atol; older code may still use tol.
        return spla.cg(linear_operator, rhs, rtol=tol, maxiter=maxiter)


def _compute_newton_direction(
    problem: QPProblem,
    x: np.ndarray,
    z: np.ndarray,
    r_dual: np.ndarray,
    r_pri: np.ndarray,
    r_cent: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, a = problem.Q, problem.A

    # H = Q + X^{-1} Z
    h = q + np.diag(z / x)

    # rhs for reduced KKT:
    # (Q + X^{-1}Z) dx - A^T dy = -r_dual - X^{-1}r_cent
    # A dx = -r_pri
    rhs1 = -r_dual - r_cent / x
    rhs2 = -r_pri

    h_inv_rhs1 = _solve_spd(h, rhs1, regularization=regularization)
    h_inv_at = _solve_spd(h, a.T, regularization=regularization)

    schur = a @ h_inv_at
    schur_rhs = rhs2 - a @ h_inv_rhs1
    dy = _solve_spd(schur, schur_rhs, regularization=regularization)
    dx = h_inv_rhs1 + h_inv_at @ dy
    dz = (-r_cent - z * dx) / x
    return dx, dy, dz


def _compute_newton_direction_sparse(
    q_sparse,
    a_sparse,
    x: np.ndarray,
    z: np.ndarray,
    r_dual: np.ndarray,
    r_pri: np.ndarray,
    r_cent: np.ndarray,
    regularization: float,
    h_solver: str,
    schur_solver: str,
    cg_tol: float,
    cg_max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    m = r_pri.shape[0]

    # H = Q + X^{-1} Z + delta*I (sparse SPD)
    h = q_sparse + sp.diags(z / x, offsets=0, shape=(n, n), format="csc")
    if regularization > 0.0:
        h = h + regularization * sp.eye(n, format="csc")

    solve_h = _build_sparse_linear_solver(h, method=h_solver, cg_tol=cg_tol, cg_max_iters=cg_max_iters)

    rhs1 = -r_dual - r_cent / x
    rhs2 = -r_pri

    h_inv_rhs1 = solve_h(rhs1)

    if schur_solver == "cg":
        def schur_matvec(v: np.ndarray) -> np.ndarray:
            at_v = _as_float_vector(a_sparse.T @ v)
            h_inv_at_v = solve_h(at_v)
            return _as_float_vector(a_sparse @ h_inv_at_v)

        schur_op = spla.LinearOperator((m, m), matvec=schur_matvec, dtype=float)
        schur_rhs = rhs2 - _as_float_vector(a_sparse @ h_inv_rhs1)
        dy, info = _cg_solve_compat(schur_op, schur_rhs, cg_tol, cg_max_iters)
        if info != 0:
            raise RuntimeError(f"CG failed for Schur system (info={info}).")
    elif schur_solver == "dense":
        a_t_dense = a_sparse.T.toarray()
        h_inv_at = solve_h(a_t_dense)
        schur = _as_float_array(a_sparse @ h_inv_at)
        schur_rhs = rhs2 - _as_float_vector(a_sparse @ h_inv_rhs1)
        dy = _solve_spd(schur, schur_rhs, regularization=regularization)
    else:
        raise ValueError("schur_solver must be one of: 'cg', 'dense'.")

    at_dy = _as_float_vector(a_sparse.T @ dy)
    dx = h_inv_rhs1 + solve_h(at_dy)
    dz = (-r_cent - z * dx) / x
    return dx, dy, dz


def primal_dual_interior_point_qp(problem: QPProblem, options: Optional[SolverOptions] = None) -> SolverResult:
    if options is None:
        options = SolverOptions()

    q = _as_float_array(problem.Q)
    c = _as_float_array(problem.c).reshape(-1)
    a = _as_float_array(problem.A)
    b = _as_float_array(problem.b).reshape(-1)

    n = c.shape[0]
    m = b.shape[0]
    if q.shape != (n, n):
        raise ValueError("Q must have shape (n, n).")
    if a.shape != (m, n):
        raise ValueError("A must have shape (m, n).")

    x = np.ones(n)
    z = np.ones(n)
    y = np.zeros(m)

    converged = False
    final_iter = 0
    pri_norm = np.inf
    dual_norm = np.inf
    mu = np.inf

    for it in range(1, options.max_iters + 1):
        r_dual = q @ x + c - a.T @ y - z
        r_pri = a @ x - b
        mu = float(np.dot(x, z) / n)

        pri_norm = float(np.linalg.norm(r_pri))
        dual_norm = float(np.linalg.norm(r_dual))
        if options.verbose:
            print(
                f"[iter={it:02d}] ||r_pri||={pri_norm:.3e} "
                f"||r_dual||={dual_norm:.3e} mu={mu:.3e}"
            )

        if pri_norm <= options.tol and dual_norm <= options.tol and mu <= options.tol:
            converged = True
            final_iter = it - 1
            break

        # Mehrotra predictor step (sigma = 0)
        r_cent_aff = x * z
        dx_aff, _, dz_aff = _compute_newton_direction(
            QPProblem(Q=q, c=c, A=a, b=b),
            x,
            z,
            r_dual,
            r_pri,
            r_cent_aff,
            regularization=options.regularization,
        )

        alpha_aff_pri = _max_step_to_positive(x, dx_aff, options.alpha_safety)
        alpha_aff_dual = _max_step_to_positive(z, dz_aff, options.alpha_safety)
        x_aff = x + alpha_aff_pri * dx_aff
        z_aff = z + alpha_aff_dual * dz_aff
        mu_aff = float(np.dot(x_aff, z_aff) / n)
        sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0

        # Corrector: X dz + Z dx = -Xz - dx_aff*dz_aff + sigma*mu*e
        r_cent = x * z + dx_aff * dz_aff - sigma * mu * np.ones(n)
        dx, dy, dz = _compute_newton_direction(
            QPProblem(Q=q, c=c, A=a, b=b),
            x,
            z,
            r_dual,
            r_pri,
            r_cent,
            regularization=options.regularization,
        )

        alpha_pri = _max_step_to_positive(x, dx, options.alpha_safety)
        alpha_dual = _max_step_to_positive(z, dz, options.alpha_safety)

        x = x + alpha_pri * dx
        y = y + alpha_dual * dy
        z = z + alpha_dual * dz
        final_iter = it

    objective = 0.5 * float(x @ q @ x) + float(c @ x)
    return SolverResult(
        x=x,
        y=y,
        z=z,
        objective=objective,
        converged=converged,
        iterations=final_iter,
        primal_residual=pri_norm,
        dual_residual=dual_norm,
        complementarity=mu,
    )


def solve_lp(c: np.ndarray, A: np.ndarray, b: np.ndarray, options: Optional[SolverOptions] = None) -> SolverResult:
    c = _as_float_array(c).reshape(-1)
    n = c.shape[0]
    q = np.zeros((n, n), dtype=float)
    return primal_dual_interior_point_qp(QPProblem(q, c, A, b), options=options)


def solve_qp(
    Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, options: Optional[SolverOptions] = None
) -> SolverResult:
    return primal_dual_interior_point_qp(QPProblem(Q, c, A, b), options=options)


def primal_dual_interior_point_qp_sparse(
    problem: QPProblem,
    options: Optional[SolverOptions] = None,
    h_solver: str = "splu",
    schur_solver: str = "cg",
    cg_tol: float = 1e-10,
    cg_max_iters: int = 400,
) -> SolverResult:
    _require_scipy()
    if options is None:
        options = SolverOptions()

    q_sparse = problem.Q if sp.issparse(problem.Q) else sp.csc_matrix(_as_float_array(problem.Q))
    a_sparse = problem.A if sp.issparse(problem.A) else sp.csc_matrix(_as_float_array(problem.A))
    q_sparse = q_sparse.tocsc()
    a_sparse = a_sparse.tocsc()

    c = _as_float_vector(problem.c)
    b = _as_float_vector(problem.b)

    n = c.shape[0]
    m = b.shape[0]
    if q_sparse.shape != (n, n):
        raise ValueError("Q must have shape (n, n).")
    if a_sparse.shape != (m, n):
        raise ValueError("A must have shape (m, n).")

    x = np.ones(n)
    z = np.ones(n)
    y = np.zeros(m)

    converged = False
    final_iter = 0
    pri_norm = np.inf
    dual_norm = np.inf
    mu = np.inf

    for it in range(1, options.max_iters + 1):
        r_dual = _as_float_vector(q_sparse @ x) + c - _as_float_vector(a_sparse.T @ y) - z
        r_pri = _as_float_vector(a_sparse @ x) - b
        mu = float(np.dot(x, z) / n)

        pri_norm = float(np.linalg.norm(r_pri))
        dual_norm = float(np.linalg.norm(r_dual))
        if options.verbose:
            print(
                f"[sparse iter={it:02d}] ||r_pri||={pri_norm:.3e} "
                f"||r_dual||={dual_norm:.3e} mu={mu:.3e}"
            )

        if pri_norm <= options.tol and dual_norm <= options.tol and mu <= options.tol:
            converged = True
            final_iter = it - 1
            break

        r_cent_aff = x * z
        dx_aff, _, dz_aff = _compute_newton_direction_sparse(
            q_sparse=q_sparse,
            a_sparse=a_sparse,
            x=x,
            z=z,
            r_dual=r_dual,
            r_pri=r_pri,
            r_cent=r_cent_aff,
            regularization=options.regularization,
            h_solver=h_solver,
            schur_solver=schur_solver,
            cg_tol=cg_tol,
            cg_max_iters=cg_max_iters,
        )

        alpha_aff_pri = _max_step_to_positive(x, dx_aff, options.alpha_safety)
        alpha_aff_dual = _max_step_to_positive(z, dz_aff, options.alpha_safety)
        x_aff = x + alpha_aff_pri * dx_aff
        z_aff = z + alpha_aff_dual * dz_aff
        mu_aff = float(np.dot(x_aff, z_aff) / n)
        sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0

        r_cent = x * z + dx_aff * dz_aff - sigma * mu * np.ones(n)
        dx, dy, dz = _compute_newton_direction_sparse(
            q_sparse=q_sparse,
            a_sparse=a_sparse,
            x=x,
            z=z,
            r_dual=r_dual,
            r_pri=r_pri,
            r_cent=r_cent,
            regularization=options.regularization,
            h_solver=h_solver,
            schur_solver=schur_solver,
            cg_tol=cg_tol,
            cg_max_iters=cg_max_iters,
        )

        alpha_pri = _max_step_to_positive(x, dx, options.alpha_safety)
        alpha_dual = _max_step_to_positive(z, dz, options.alpha_safety)

        x = x + alpha_pri * dx
        y = y + alpha_dual * dy
        z = z + alpha_dual * dz
        final_iter = it

    objective = 0.5 * float(x @ (_as_float_array(q_sparse @ x))) + float(c @ x)
    return SolverResult(
        x=x,
        y=y,
        z=z,
        objective=objective,
        converged=converged,
        iterations=final_iter,
        primal_residual=pri_norm,
        dual_residual=dual_norm,
        complementarity=mu,
    )


def solve_lp_sparse(
    c: np.ndarray,
    A,
    b: np.ndarray,
    options: Optional[SolverOptions] = None,
    h_solver: str = "splu",
    schur_solver: str = "cg",
    cg_tol: float = 1e-10,
    cg_max_iters: int = 400,
) -> SolverResult:
    _require_scipy()
    c_vec = _as_float_vector(c)
    n = c_vec.shape[0]
    q = sp.csc_matrix((n, n), dtype=float)
    return primal_dual_interior_point_qp_sparse(
        QPProblem(q, c_vec, A, b),
        options=options,
        h_solver=h_solver,
        schur_solver=schur_solver,
        cg_tol=cg_tol,
        cg_max_iters=cg_max_iters,
    )


def solve_qp_sparse(
    Q,
    c: np.ndarray,
    A,
    b: np.ndarray,
    options: Optional[SolverOptions] = None,
    h_solver: str = "splu",
    schur_solver: str = "cg",
    cg_tol: float = 1e-10,
    cg_max_iters: int = 400,
) -> SolverResult:
    return primal_dual_interior_point_qp_sparse(
        QPProblem(Q, c, A, b),
        options=options,
        h_solver=h_solver,
        schur_solver=schur_solver,
        cg_tol=cg_tol,
        cg_max_iters=cg_max_iters,
    )

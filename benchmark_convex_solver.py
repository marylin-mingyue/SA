import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import scipy.sparse as sp
except Exception as exc:
    raise ImportError(
        "benchmark_convex_solver.py requires scipy. Install via `python3 -m pip install scipy`."
    ) from exc

from mini_convex_solver import SolverOptions, solve_qp, solve_qp_sparse


def _parse_sizes(raw: str) -> List[int]:
    sizes: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value <= 1:
                raise ValueError("Each size in --sizes must be > 1.")
            sizes.append(value)
    if not sizes:
        raise ValueError("--sizes must contain at least one integer.")
    return sizes


def _generate_sparse_qp_instance(
    n: int,
    m: int,
    density: float,
    seed: int,
) -> Tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_star = rng.uniform(0.2, 1.2, size=n)

    A = sp.random(
        m,
        n,
        density=density,
        format="csc",
        random_state=seed,
        data_rvs=lambda k: rng.uniform(0.2, 1.0, size=k),
    )
    b = np.asarray(A @ x_star).reshape(-1)

    q_diag = rng.uniform(0.5, 2.0, size=n)
    Q = sp.diags(q_diag, offsets=0, format="csc")
    y_star = rng.normal(0.0, 0.1, size=m)
    c = -(q_diag * x_star) + np.asarray(A.T @ y_star).reshape(-1)
    return Q, c, A, b, x_star


def _run_dense_trial(
    Q_sparse: sp.csc_matrix,
    c: np.ndarray,
    A_sparse: sp.csc_matrix,
    b: np.ndarray,
    x_star: np.ndarray,
    options: SolverOptions,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    result = solve_qp(Q_sparse.toarray(), c, A_sparse.toarray(), b, options=options)
    elapsed = time.perf_counter() - t0

    return {
        "time_sec": elapsed,
        "converged": 1.0 if result.converged else 0.0,
        "iterations": float(result.iterations),
        "primal_residual": float(result.primal_residual),
        "dual_residual": float(result.dual_residual),
        "complementarity": float(result.complementarity),
        "x_error_l2": float(np.linalg.norm(result.x - x_star)),
        "objective": float(result.objective),
    }


def _run_sparse_trial(
    Q_sparse: sp.csc_matrix,
    c: np.ndarray,
    A_sparse: sp.csc_matrix,
    b: np.ndarray,
    x_star: np.ndarray,
    options: SolverOptions,
    h_solver: str,
    schur_solver: str,
    cg_tol: float,
    cg_max_iters: int,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    result = solve_qp_sparse(
        Q_sparse,
        c,
        A_sparse,
        b,
        options=options,
        h_solver=h_solver,
        schur_solver=schur_solver,
        cg_tol=cg_tol,
        cg_max_iters=cg_max_iters,
    )
    elapsed = time.perf_counter() - t0

    return {
        "time_sec": elapsed,
        "converged": 1.0 if result.converged else 0.0,
        "iterations": float(result.iterations),
        "primal_residual": float(result.primal_residual),
        "dual_residual": float(result.dual_residual),
        "complementarity": float(result.complementarity),
        "x_error_l2": float(np.linalg.norm(result.x - x_star)),
        "objective": float(result.objective),
    }


def _summarize(group: List[Dict[str, float]]) -> Dict[str, float]:
    if not group:
        return {}
    return {
        "runs": float(len(group)),
        "success_rate": mean(v["converged"] for v in group),
        "time_mean_sec": mean(v["time_sec"] for v in group),
        "time_std_sec": pstdev(v["time_sec"] for v in group) if len(group) > 1 else 0.0,
        "iterations_mean": mean(v["iterations"] for v in group),
        "x_error_l2_mean": mean(v["x_error_l2"] for v in group),
        "primal_residual_mean": mean(v["primal_residual"] for v in group),
        "dual_residual_mean": mean(v["dual_residual"] for v in group),
        "complementarity_mean": mean(v["complementarity"] for v in group),
    }


def _write_summary_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "n",
        "m",
        "solver",
        "runs",
        "success_rate",
        "time_mean_sec",
        "time_std_sec",
        "iterations_mean",
        "x_error_l2_mean",
        "primal_residual_mean",
        "dual_residual_mean",
        "complementarity_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_size_time(summary_rows: List[Dict[str, float]], output_path: Path) -> None:
    if not summary_rows:
        return

    by_solver: Dict[str, List[Tuple[float, float]]] = {}
    for row in summary_rows:
        solver = str(row["solver"])
        by_solver.setdefault(solver, []).append((float(row["n"]), float(row["time_mean_sec"])))

    plt.figure(figsize=(8.8, 5.2))
    solver_order = ["dense", "sparse"]
    for solver in solver_order:
        if solver not in by_solver:
            continue
        points = sorted(by_solver[solver], key=lambda p: p[0])
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        plt.plot(x_vals, y_vals, marker="o", linewidth=2, label=solver)

    for solver, points in by_solver.items():
        if solver in solver_order:
            continue
        points = sorted(points, key=lambda p: p[0])
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        plt.plot(x_vals, y_vals, marker="o", linewidth=2, label=solver)

    plt.xlabel("Problem size n")
    plt.ylabel("Mean time (sec)")
    plt.title("Size-Time Curve (Dense vs Sparse)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_speedup(summary_rows: List[Dict[str, float]], output_path: Path) -> None:
    dense_by_n: Dict[float, float] = {}
    sparse_by_n: Dict[float, float] = {}
    for row in summary_rows:
        n = float(row["n"])
        time_sec = float(row["time_mean_sec"])
        if row["solver"] == "dense":
            dense_by_n[n] = time_sec
        elif row["solver"] == "sparse":
            sparse_by_n[n] = time_sec

    common_n = sorted(set(dense_by_n.keys()) & set(sparse_by_n.keys()))
    plt.figure(figsize=(8.8, 5.2))
    if common_n:
        speedups = [dense_by_n[n] / sparse_by_n[n] for n in common_n]
        plt.plot(common_n, speedups, marker="o", linewidth=2, color="#d62728")
        plt.axhline(1.0, linestyle="--", linewidth=1.2, color="#555555")
        plt.ylabel("Speedup (dense_time / sparse_time)")
    else:
        plt.text(0.5, 0.5, "No common n with both dense and sparse results", ha="center", va="center")
        plt.ylabel("Speedup")

    plt.xlabel("Problem size n")
    plt.title("Sparse-vs-Dense Speedup Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dense vs sparse primal-dual interior-point solver.")
    parser.add_argument(
        "--sizes",
        type=str,
        default="80,140,220,320",
        help="Comma-separated problem sizes n (variables).",
    )
    parser.add_argument("--constraint-ratio", type=float, default=0.2, help="m = max(1, ratio*n).")
    parser.add_argument("--density", type=float, default=0.02, help="Sparse density for A.")
    parser.add_argument("--trials", type=int, default=3, help="Trials per size.")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument(
        "--dense-max-n",
        type=int,
        default=220,
        help="Run dense baseline only when n <= dense-max-n.",
    )
    parser.add_argument("--output-dir", type=str, default="benchmarks", help="Output folder.")
    parser.add_argument("--max-iters", type=int, default=80, help="Solver max iterations.")
    parser.add_argument("--tol", type=float, default=1e-8, help="Solver stopping tolerance.")
    parser.add_argument("--alpha-safety", type=float, default=0.995, help="Interior-point step safety.")
    parser.add_argument("--regularization", type=float, default=1e-10, help="Linear-system regularization.")
    parser.add_argument(
        "--sparse-h-solver",
        type=str,
        default="splu",
        choices=["splu", "cg"],
        help="Sparse solver for H-system.",
    )
    parser.add_argument(
        "--sparse-schur-solver",
        type=str,
        default="cg",
        choices=["cg", "dense"],
        help="Sparse solver for Schur complement system.",
    )
    parser.add_argument("--sparse-cg-tol", type=float, default=1e-9, help="CG tolerance for sparse mode.")
    parser.add_argument("--sparse-cg-max-iters", type=int, default=800, help="CG max iterations for sparse mode.")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating benchmark curve images.",
    )
    args = parser.parse_args()
    if not args.skip_plots and not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Plotting requires matplotlib. Install it with `python3 -m pip install matplotlib` "
            "or run with --skip-plots."
        )

    sizes = _parse_sizes(args.sizes)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    options = SolverOptions(
        max_iters=args.max_iters,
        tol=args.tol,
        alpha_safety=args.alpha_safety,
        regularization=args.regularization,
        verbose=False,
    )

    raw_records: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []

    print("Running benchmark...")
    for idx, n in enumerate(sizes):
        m = max(1, int(round(args.constraint_ratio * n)))
        dense_trials: List[Dict[str, float]] = []
        sparse_trials: List[Dict[str, float]] = []

        for trial in range(args.trials):
            seed = args.seed + idx * 10000 + trial * 17
            Q, c, A, b, x_star = _generate_sparse_qp_instance(n, m, args.density, seed)

            sparse = _run_sparse_trial(
                Q,
                c,
                A,
                b,
                x_star,
                options=options,
                h_solver=args.sparse_h_solver,
                schur_solver=args.sparse_schur_solver,
                cg_tol=args.sparse_cg_tol,
                cg_max_iters=args.sparse_cg_max_iters,
            )
            sparse_trials.append(sparse)
            raw_records.append({"n": float(n), "m": float(m), "solver": "sparse", "trial": float(trial), **sparse})

            if n <= args.dense_max_n:
                dense = _run_dense_trial(Q, c, A, b, x_star, options=options)
                dense_trials.append(dense)
                raw_records.append(
                    {"n": float(n), "m": float(m), "solver": "dense", "trial": float(trial), **dense}
                )

        dense_summary = _summarize(dense_trials)
        sparse_summary = _summarize(sparse_trials)

        if dense_summary:
            summary_rows.append({"n": n, "m": m, "solver": "dense", **dense_summary})
        summary_rows.append({"n": n, "m": m, "solver": "sparse", **sparse_summary})

        dense_time = dense_summary.get("time_mean_sec")
        sparse_time = sparse_summary.get("time_mean_sec")
        speedup = (dense_time / sparse_time) if (dense_time and sparse_time) else None
        speedup_text = f"{speedup:.2f}x" if speedup is not None else "N/A (dense skipped)"
        print(
            f"  n={n:4d}, m={m:4d} | "
            f"sparse={sparse_time:.4f}s | "
            f"dense={(f'{dense_time:.4f}s' if dense_time is not None else 'skipped')} | "
            f"speedup={speedup_text}"
        )

    summary_csv = out_dir / "solver_benchmark_summary.csv"
    raw_json = out_dir / "solver_benchmark_raw.json"
    summary_json = out_dir / "solver_benchmark_summary.json"
    size_time_plot = out_dir / "solver_benchmark_size_time.png"
    speedup_plot = out_dir / "solver_benchmark_speedup.png"

    _write_summary_csv(summary_csv, summary_rows)
    raw_json.write_text(json.dumps(raw_records, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    if not args.skip_plots:
        _plot_size_time(summary_rows, size_time_plot)
        _plot_speedup(summary_rows, speedup_plot)

    print("\nSaved benchmark reports:")
    print(f"  - {summary_csv}")
    print(f"  - {summary_json}")
    print(f"  - {raw_json}")
    if not args.skip_plots:
        print(f"  - {size_time_plot}")
        print(f"  - {speedup_plot}")


if __name__ == "__main__":
    main()

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch

from train_mlp_compare import (
    History,
    MLP,
    first_epoch_below,
    make_nonlinear_dataset,
    train_with_gauss_newton,
    train_with_sgd,
    train_with_trust_region,
    train_with_trust_region_gauss_newton,
)


def run_experiments(
    epochs: int,
    model_seed: int,
    dataset_seed: int,
    hidden_size: int,
    threshold: float,
) -> Tuple[Dict[str, History], Dict[str, float]]:
    torch.manual_seed(model_seed)
    x, y = make_nonlinear_dataset(n_samples=2048, seed=dataset_seed)

    base_model = MLP(hidden=hidden_size)
    model_sgd = copy.deepcopy(base_model)
    model_trn = copy.deepcopy(base_model)
    model_gn = copy.deepcopy(base_model)
    model_trgn = copy.deepcopy(base_model)

    histories: Dict[str, History] = {
        "SGD": train_with_sgd(model_sgd, x, y, epochs=epochs, lr=0.15),
        "TrustRegionNewtonCG": train_with_trust_region(
            model_trn, x, y, epochs=epochs, trust_radius=1.0
        ),
        "GaussNewtonCG": train_with_gauss_newton(model_gn, x, y, epochs=epochs),
        "TrustRegionGaussNewtonCG": train_with_trust_region_gauss_newton(
            model_trgn, x, y, epochs=epochs, trust_radius=1.0
        ),
    }

    first_hit = {
        name: first_epoch_below(history.losses, threshold)
        for name, history in histories.items()
    }
    return histories, first_hit


def plot_loss_curves(histories: Dict[str, History], output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    epochs_axis = range(1, len(next(iter(histories.values())).losses) + 1)
    for name, history in histories.items():
        plt.plot(epochs_axis, history.losses, label=name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_accuracy_curves(histories: Dict[str, History], output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    epochs_axis = range(1, len(next(iter(histories.values())).accs) + 1)
    for name, history in histories.items():
        accuracies = [v * 100.0 for v in history.accs]
        plt.plot(epochs_axis, accuracies, label=name, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_threshold_epochs(first_hit: Dict[str, int], total_epochs: int, output_path: Path) -> None:
    names = list(first_hit.keys())
    raw_values = [first_hit[name] for name in names]
    bar_values = [value if value != -1 else total_epochs + 2 for value in raw_values]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, bar_values, alpha=0.85)
    plt.ylabel("Epoch")
    plt.title("Epoch to Reach Target Loss Threshold")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=12)

    for bar, raw in zip(bars, raw_values):
        label = str(raw) if raw != -1 else "N/A"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def dump_metrics(
    histories: Dict[str, History],
    first_hit: Dict[str, int],
    threshold: float,
    output_path: Path,
) -> None:
    payload = {
        "threshold": threshold,
        "methods": {},
    }
    for name, history in histories.items():
        payload["methods"][name] = {
            "final_loss": history.losses[-1],
            "final_accuracy": history.accs[-1],
            "loss_at_10": history.losses[9] if len(history.losses) >= 10 else None,
            "loss_at_20": history.losses[19] if len(history.losses) >= 20 else None,
            "epoch_to_threshold": first_hit[name],
        }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimization experiments and generate analysis plots.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for each optimizer.")
    parser.add_argument("--threshold", type=float, default=0.03, help="Target loss threshold.")
    parser.add_argument("--model-seed", type=int, default=42, help="Random seed for model initialization.")
    parser.add_argument("--dataset-seed", type=int, default=7, help="Random seed for synthetic dataset.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer width in MLP.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to store generated figures and metrics.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    histories, first_hit = run_experiments(
        epochs=args.epochs,
        model_seed=args.model_seed,
        dataset_seed=args.dataset_seed,
        hidden_size=args.hidden_size,
        threshold=args.threshold,
    )

    loss_fig = out_dir / "loss_curve.png"
    acc_fig = out_dir / "accuracy_curve.png"
    epoch_fig = out_dir / "epoch_to_threshold.png"
    metrics_json = out_dir / "metrics.json"

    plot_loss_curves(histories, loss_fig)
    plot_accuracy_curves(histories, acc_fig)
    plot_threshold_epochs(first_hit, args.epochs, epoch_fig)
    dump_metrics(histories, first_hit, args.threshold, metrics_json)

    print("Generated:")
    print(f"  - {loss_fig}")
    print(f"  - {acc_fig}")
    print(f"  - {epoch_fig}")
    print(f"  - {metrics_json}")


if __name__ == "__main__":
    main()

import copy
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from second_order_optimizer import (
    GaussNewtonCG,
    TrustRegionGaussNewtonCG,
    TrustRegionNewtonCG,
)


def make_nonlinear_dataset(n_samples: int = 2048, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = 2.0 * torch.rand(n_samples, 2, generator=g) - 1.0
    # Nonlinear binary boundary.
    score = x[:, 0] * x[:, 1] + 0.6 * x[:, 0] * x[:, 0] - 0.25
    y = (score > 0.0).float().unsqueeze(1)
    return x, y


class MLP(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class History:
    losses: List[float]
    accs: List[float]


def mse_half_loss(logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    residual = logits - y
    loss = 0.5 * torch.mean(residual * residual)
    return loss, residual


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        logits = model(x)
        loss, _ = mse_half_loss(logits, y)
        preds = (logits > 0.5).float()
        acc = (preds == y).float().mean().item()
    return loss.item(), acc


def train_with_sgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
) -> History:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    losses: List[float] = []
    accs: List[float] = []

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, _ = mse_half_loss(logits, y)
        loss.backward()
        optimizer.step()
        l, a = evaluate(model, x, y)
        losses.append(l)
        accs.append(a)
    return History(losses=losses, accs=accs)


def train_with_trust_region(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    trust_radius: float,
) -> History:
    optimizer = TrustRegionNewtonCG(
        model.parameters(),
        trust_radius=trust_radius,
        max_trust_radius=20.0,
        acceptance_eta=0.1,
        cg_max_iters=30,
        cg_tol=1e-6,
        damping=5e-2,
    )
    losses: List[float] = []
    accs: List[float] = []

    for _ in range(epochs):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, _ = mse_half_loss(logits, y)
            return loss

        _ = optimizer.step(closure)
        l, a = evaluate(model, x, y)
        losses.append(l)
        accs.append(a)
    return History(losses=losses, accs=accs)


def train_with_gauss_newton(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
) -> History:
    optimizer = GaussNewtonCG(
        model.parameters(),
        lr=1.0,
        damping=5e-2,
        cg_max_iters=30,
        cg_tol=1e-6,
        backtrack_factor=0.5,
        min_step_scale=1e-4,
    )
    losses: List[float] = []
    accs: List[float] = []

    for _ in range(epochs):
        def closure() -> Tuple[torch.Tensor, torch.Tensor]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, residual = mse_half_loss(logits, y)
            return loss, residual

        _ = optimizer.step(closure)
        l, a = evaluate(model, x, y)
        losses.append(l)
        accs.append(a)
    return History(losses=losses, accs=accs)


def train_with_trust_region_gauss_newton(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    trust_radius: float,
) -> History:
    optimizer = TrustRegionGaussNewtonCG(
        model.parameters(),
        trust_radius=trust_radius,
        max_trust_radius=20.0,
        acceptance_eta=0.1,
        cg_max_iters=30,
        cg_tol=1e-6,
        damping=5e-2,
    )
    losses: List[float] = []
    accs: List[float] = []

    for _ in range(epochs):
        def closure() -> Tuple[torch.Tensor, torch.Tensor]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, residual = mse_half_loss(logits, y)
            return loss, residual

        _ = optimizer.step(closure)
        l, a = evaluate(model, x, y)
        losses.append(l)
        accs.append(a)
    return History(losses=losses, accs=accs)


def first_epoch_below(losses: List[float], threshold: float) -> int:
    for i, v in enumerate(losses, start=1):
        if v <= threshold:
            return i
    return -1


def report_result(name: str, h: History) -> None:
    print(f"{name}:")
    print(f"  final loss: {h.losses[-1]:.6f}")
    print(f"  final acc : {h.accs[-1] * 100:.2f}%")
    print(f"  loss@10   : {h.losses[9]:.6f}")
    print(f"  loss@20   : {h.losses[19]:.6f}")
    print()


def main() -> None:
    torch.manual_seed(42)
    x, y = make_nonlinear_dataset(n_samples=2048, seed=7)

    base_model = MLP(hidden=64)
    model_sgd = copy.deepcopy(base_model)
    model_tr = copy.deepcopy(base_model)
    model_gn = copy.deepcopy(base_model)
    model_trgn = copy.deepcopy(base_model)

    epochs = 50
    sgd_hist = train_with_sgd(model_sgd, x, y, epochs=epochs, lr=0.15)
    tr_hist = train_with_trust_region(model_tr, x, y, epochs=epochs, trust_radius=1.0)
    gn_hist = train_with_gauss_newton(model_gn, x, y, epochs=epochs)
    trgn_hist = train_with_trust_region_gauss_newton(
        model_trgn, x, y, epochs=epochs, trust_radius=1.0
    )

    report_result("SGD (momentum=0.9)", sgd_hist)
    report_result("TrustRegionNewtonCG", tr_hist)
    report_result("GaussNewtonCG", gn_hist)
    report_result("TrustRegionGaussNewtonCG", trgn_hist)

    threshold = 0.03
    sgd_ep = first_epoch_below(sgd_hist.losses, threshold)
    tr_ep = first_epoch_below(tr_hist.losses, threshold)
    gn_ep = first_epoch_below(gn_hist.losses, threshold)
    trgn_ep = first_epoch_below(trgn_hist.losses, threshold)
    print(f"Epoch to reach loss <= {threshold}:")
    print(f"  SGD               : {sgd_ep if sgd_ep != -1 else 'not reached'}")
    print(f"  TrustRegionNewton : {tr_ep if tr_ep != -1 else 'not reached'}")
    print(f"  GaussNewton       : {gn_ep if gn_ep != -1 else 'not reached'}")
    print(f"  TR-GaussNewton    : {trgn_ep if trgn_ep != -1 else 'not reached'}")


if __name__ == "__main__":
    main()

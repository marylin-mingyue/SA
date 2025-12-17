import math
from typing import Callable, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


def _flatten_tensors(tensors: Sequence[Tensor]) -> Tensor:
    if not tensors:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat([t.reshape(-1) for t in tensors])


def _vector_to_shapes(vector: Tensor, params: Sequence[Tensor]) -> List[Tensor]:
    chunks: List[Tensor] = []
    offset = 0
    for p in params:
        numel = p.numel()
        chunks.append(vector[offset : offset + numel].view_as(p))
        offset += numel
    return chunks


def _gather_params(params: Iterable[Tensor]) -> List[Tensor]:
    return [p for p in params if p.requires_grad]


class TrustRegionNewtonCG(Optimizer):
    """Second-order optimizer using trust-region + Steihaug conjugate gradient.

    Key features:
    - Hessian-vector product via autograd (no explicit Hessian construction).
    - Damping term for improved stability on non-convex problems.
    - Truncated CG that detects negative curvature and respects trust radius.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        trust_radius: float = 1.0,
        max_trust_radius: float = 100.0,
        acceptance_eta: float = 0.1,
        cg_max_iters: int = 25,
        cg_tol: float = 1e-6,
        damping: float = 1e-3,
    ) -> None:
        defaults = dict(
            trust_radius=trust_radius,
            max_trust_radius=max_trust_radius,
            acceptance_eta=acceptance_eta,
            cg_max_iters=cg_max_iters,
            cg_tol=cg_tol,
            damping=damping,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("TrustRegionNewtonCG currently supports exactly one param group.")

    @torch.no_grad()
    def _apply_step(self, params: Sequence[Tensor], step: Tensor, scale: float = 1.0) -> None:
        chunks = _vector_to_shapes(step, params)
        for p, delta in zip(params, chunks):
            p.add_(delta, alpha=scale)

    def _calc_tau_to_boundary(self, z: Tensor, d: Tensor, radius: float) -> float:
        # Solve ||z + tau*d||^2 = radius^2 for tau > 0.
        a = torch.dot(d, d).item()
        b = 2.0 * torch.dot(z, d).item()
        c = torch.dot(z, z).item() - radius * radius
        disc = max(b * b - 4.0 * a * c, 0.0)
        tau = (-b + math.sqrt(disc)) / (2.0 * a)
        return tau

    def _steihaug_cg(
        self,
        grad: Tensor,
        hvp_fn: Callable[[Tensor], Tensor],
        radius: float,
        cg_max_iters: int,
        cg_tol: float,
    ) -> Tuple[Tensor, bool, int]:
        z = torch.zeros_like(grad)
        r = grad.clone()
        d = -r
        hit_boundary = False

        if r.norm().item() < cg_tol:
            return z, hit_boundary, 0

        for k in range(cg_max_iters):
            Bd = hvp_fn(d)
            dBd = torch.dot(d, Bd).item()

            if dBd <= 0.0:
                tau = self._calc_tau_to_boundary(z, d, radius)
                z = z + tau * d
                hit_boundary = True
                return z, hit_boundary, k + 1

            rr = torch.dot(r, r).item()
            alpha = rr / dBd
            z_next = z + alpha * d

            if z_next.norm().item() >= radius:
                tau = self._calc_tau_to_boundary(z, d, radius)
                z = z + tau * d
                hit_boundary = True
                return z, hit_boundary, k + 1

            r_next = r + alpha * Bd
            if r_next.norm().item() < cg_tol:
                return z_next, hit_boundary, k + 1

            beta = torch.dot(r_next, r_next).item() / rr
            d = -r_next + beta * d
            z = z_next
            r = r_next

        return z, hit_boundary, cg_max_iters

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] = None) -> Tensor:
        if closure is None:
            raise ValueError("TrustRegionNewtonCG requires a closure returning the loss tensor.")

        group = self.param_groups[0]
        params = _gather_params(group["params"])
        if not params:
            raise ValueError("No trainable parameters found.")

        # 1) Build gradient graph for HVP.
        with torch.enable_grad():
            base_loss = closure()
        raw_grads = torch.autograd.grad(
            base_loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(raw_grads, params)]
        flat_grad = _flatten_tensors(grads).detach()
        grad_norm = flat_grad.norm().item()
        if grad_norm == 0.0:
            return base_loss.detach()

        damping = float(group["damping"])

        def hvp_fn(v: Tensor) -> Tensor:
            vecs = _vector_to_shapes(v, params)
            raw_hvps = torch.autograd.grad(
                grads,
                params,
                grad_outputs=vecs,
                retain_graph=True,
                allow_unused=True,
            )
            hvps = [h if h is not None else torch.zeros_like(p) for h, p in zip(raw_hvps, params)]
            flat_hvp = _flatten_tensors(hvps).detach()
            return flat_hvp + damping * v

        # 2) Solve trust-region subproblem.
        trust_radius = float(group["trust_radius"])
        step_vec, hit_boundary, _ = self._steihaug_cg(
            grad=flat_grad,
            hvp_fn=hvp_fn,
            radius=trust_radius,
            cg_max_iters=int(group["cg_max_iters"]),
            cg_tol=float(group["cg_tol"]),
        )

        # 3) Compute predicted reduction m(0)-m(p).
        Bp = hvp_fn(step_vec)
        predicted_reduction = -(
            torch.dot(flat_grad, step_vec).item() + 0.5 * torch.dot(step_vec, Bp).item()
        )
        if predicted_reduction <= 1e-16:
            group["trust_radius"] = max(1e-8, 0.25 * trust_radius)
            return base_loss.detach()

        # 4) Evaluate trial point and compute ratio rho.
        old_params = [p.detach().clone() for p in params]
        self._apply_step(params, step_vec, scale=1.0)
        with torch.enable_grad():
            trial_loss = closure().detach()

        actual_reduction = base_loss.detach().item() - trial_loss.item()
        rho = actual_reduction / predicted_reduction

        # 5) Adjust trust radius.
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hit_boundary:
            trust_radius = min(2.0 * trust_radius, float(group["max_trust_radius"]))
        group["trust_radius"] = max(1e-8, trust_radius)

        # 6) Accept or reject.
        if rho <= float(group["acceptance_eta"]):
            for p, old in zip(params, old_params):
                p.copy_(old)
            return base_loss.detach()
        return trial_loss


class GaussNewtonCG(Optimizer):
    """Damped Gauss-Newton optimizer solved by conjugate gradient.

    Closure must return a tuple: (loss, residual_vector), where residual_vector
    is typically model(x) - target for least-squares objectives.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0,
        damping: float = 1e-2,
        cg_max_iters: int = 25,
        cg_tol: float = 1e-6,
        backtrack_factor: float = 0.5,
        min_step_scale: float = 1e-4,
    ) -> None:
        defaults = dict(
            lr=lr,
            damping=damping,
            cg_max_iters=cg_max_iters,
            cg_tol=cg_tol,
            backtrack_factor=backtrack_factor,
            min_step_scale=min_step_scale,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("GaussNewtonCG currently supports exactly one param group.")

    @torch.no_grad()
    def _apply_step(self, params: Sequence[Tensor], step: Tensor, scale: float = 1.0) -> None:
        chunks = _vector_to_shapes(step, params)
        for p, delta in zip(params, chunks):
            p.add_(delta, alpha=scale)

    def _cg_solve(
        self,
        matvec: Callable[[Tensor], Tensor],
        rhs: Tensor,
        max_iters: int,
        tol: float,
    ) -> Tensor:
        x = torch.zeros_like(rhs)
        r = rhs - matvec(x)
        p = r.clone()
        rr_old = torch.dot(r, r).item()

        if math.sqrt(rr_old) < tol:
            return x

        for _ in range(max_iters):
            Ap = matvec(p)
            pAp = torch.dot(p, Ap).item()
            if pAp <= 1e-20:
                break

            alpha = rr_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = torch.dot(r, r).item()

            if math.sqrt(rr_new) < tol:
                break

            beta = rr_new / rr_old
            p = r + beta * p
            rr_old = rr_new
        return x

    @torch.no_grad()
    def step(self, closure: Callable[[], Tuple[Tensor, Tensor]] = None) -> Tensor:
        if closure is None:
            raise ValueError(
                "GaussNewtonCG requires closure returning (loss, residual_vector)."
            )

        group = self.param_groups[0]
        params = _gather_params(group["params"])
        if not params:
            raise ValueError("No trainable parameters found.")

        with torch.enable_grad():
            base_loss, residual = closure()
            residual_flat = residual.reshape(-1)
            n = residual_flat.numel()

            if n == 0:
                return base_loss.detach()

            # Use normalized residual to match mean-squared scaling.
            r_norm = residual_flat / math.sqrt(float(n))
            objective = 0.5 * torch.dot(r_norm, r_norm)
            raw_grads = torch.autograd.grad(
                objective,
                params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(raw_grads, params)]
        flat_grad = _flatten_tensors(grads).detach()
        if flat_grad.norm().item() == 0.0:
            return base_loss.detach()

        damping = float(group["damping"])

        def gn_matvec(v: Tensor) -> Tensor:
            with torch.enable_grad():
                vecs = _vector_to_shapes(v, params)

                # Jacobian-vector product: jv = Jv
                u = torch.ones_like(r_norm, requires_grad=True)
                jt_u = torch.autograd.grad(
                    r_norm,
                    params,
                    grad_outputs=u,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )
                jt_u = [g if g is not None else torch.zeros_like(p) for g, p in zip(jt_u, params)]
                dot_val = torch.tensor(0.0, device=v.device, dtype=v.dtype)
                for g, vi in zip(jt_u, vecs):
                    dot_val = dot_val + torch.sum(g * vi)
                jv = torch.autograd.grad(dot_val, u, retain_graph=True)[0].detach()

                # Vector-Jacobian product: J^T (Jv)
                jt_jv = torch.autograd.grad(
                    r_norm,
                    params,
                    grad_outputs=jv,
                    retain_graph=True,
                    allow_unused=True,
                )
                jt_jv = [h if h is not None else torch.zeros_like(p) for h, p in zip(jt_jv, params)]
                flat_jtjv = _flatten_tensors(jt_jv).detach()
                return flat_jtjv + damping * v

        rhs = -flat_grad
        delta = self._cg_solve(
            matvec=gn_matvec,
            rhs=rhs,
            max_iters=int(group["cg_max_iters"]),
            tol=float(group["cg_tol"]),
        )
        step_vec = float(group["lr"]) * delta

        old_params = [p.detach().clone() for p in params]
        step_scale = 1.0
        accepted = False
        trial_loss = base_loss.detach()

        while step_scale >= float(group["min_step_scale"]):
            self._apply_step(params, step_vec, scale=step_scale)
            with torch.enable_grad():
                candidate_loss, _ = closure()
            if candidate_loss.detach().item() <= base_loss.detach().item():
                accepted = True
                trial_loss = candidate_loss.detach()
                break

            for p, old in zip(params, old_params):
                p.copy_(old)
            step_scale *= float(group["backtrack_factor"])

        if accepted:
            group["damping"] = max(1e-6, damping * 0.7)
            return trial_loss

        group["damping"] = min(1e3, damping * 2.0)
        return base_loss.detach()


class TrustRegionGaussNewtonCG(Optimizer):
    """Trust-region Gauss-Newton optimizer with rho-based acceptance.

    Closure must return a tuple: (loss, residual_vector), where residual_vector
    corresponds to a least-squares form loss ~= 0.5 * ||residual||^2.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        trust_radius: float = 1.0,
        max_trust_radius: float = 100.0,
        acceptance_eta: float = 0.1,
        cg_max_iters: int = 25,
        cg_tol: float = 1e-6,
        damping: float = 1e-2,
    ) -> None:
        defaults = dict(
            trust_radius=trust_radius,
            max_trust_radius=max_trust_radius,
            acceptance_eta=acceptance_eta,
            cg_max_iters=cg_max_iters,
            cg_tol=cg_tol,
            damping=damping,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("TrustRegionGaussNewtonCG currently supports exactly one param group.")

    @torch.no_grad()
    def _apply_step(self, params: Sequence[Tensor], step: Tensor, scale: float = 1.0) -> None:
        chunks = _vector_to_shapes(step, params)
        for p, delta in zip(params, chunks):
            p.add_(delta, alpha=scale)

    def _calc_tau_to_boundary(self, z: Tensor, d: Tensor, radius: float) -> float:
        a = torch.dot(d, d).item()
        b = 2.0 * torch.dot(z, d).item()
        c = torch.dot(z, z).item() - radius * radius
        disc = max(b * b - 4.0 * a * c, 0.0)
        return (-b + math.sqrt(disc)) / (2.0 * a)

    def _steihaug_cg(
        self,
        grad: Tensor,
        matvec: Callable[[Tensor], Tensor],
        radius: float,
        cg_max_iters: int,
        cg_tol: float,
    ) -> Tuple[Tensor, bool, int]:
        z = torch.zeros_like(grad)
        r = grad.clone()
        d = -r
        hit_boundary = False

        if r.norm().item() < cg_tol:
            return z, hit_boundary, 0

        for k in range(cg_max_iters):
            Bd = matvec(d)
            dBd = torch.dot(d, Bd).item()

            if dBd <= 0.0:
                tau = self._calc_tau_to_boundary(z, d, radius)
                z = z + tau * d
                hit_boundary = True
                return z, hit_boundary, k + 1

            rr = torch.dot(r, r).item()
            alpha = rr / dBd
            z_next = z + alpha * d

            if z_next.norm().item() >= radius:
                tau = self._calc_tau_to_boundary(z, d, radius)
                z = z + tau * d
                hit_boundary = True
                return z, hit_boundary, k + 1

            r_next = r + alpha * Bd
            if r_next.norm().item() < cg_tol:
                return z_next, hit_boundary, k + 1

            beta = torch.dot(r_next, r_next).item() / rr
            d = -r_next + beta * d
            z = z_next
            r = r_next

        return z, hit_boundary, cg_max_iters

    @torch.no_grad()
    def step(self, closure: Callable[[], Tuple[Tensor, Tensor]] = None) -> Tensor:
        if closure is None:
            raise ValueError(
                "TrustRegionGaussNewtonCG requires closure returning (loss, residual_vector)."
            )

        group = self.param_groups[0]
        params = _gather_params(group["params"])
        if not params:
            raise ValueError("No trainable parameters found.")

        with torch.enable_grad():
            base_loss, residual = closure()
            residual_flat = residual.reshape(-1)
            n = residual_flat.numel()
            if n == 0:
                return base_loss.detach()

            r_norm = residual_flat / math.sqrt(float(n))
            objective = 0.5 * torch.dot(r_norm, r_norm)
            raw_grads = torch.autograd.grad(
                objective,
                params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(raw_grads, params)]
        flat_grad = _flatten_tensors(grads).detach()
        if flat_grad.norm().item() == 0.0:
            return base_loss.detach()

        damping = float(group["damping"])

        def gn_matvec(v: Tensor) -> Tensor:
            with torch.enable_grad():
                vecs = _vector_to_shapes(v, params)

                u = torch.ones_like(r_norm, requires_grad=True)
                jt_u = torch.autograd.grad(
                    r_norm,
                    params,
                    grad_outputs=u,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )
                jt_u = [g if g is not None else torch.zeros_like(p) for g, p in zip(jt_u, params)]
                dot_val = torch.tensor(0.0, device=v.device, dtype=v.dtype)
                for g, vi in zip(jt_u, vecs):
                    dot_val = dot_val + torch.sum(g * vi)
                jv = torch.autograd.grad(dot_val, u, retain_graph=True)[0].detach()

                jt_jv = torch.autograd.grad(
                    r_norm,
                    params,
                    grad_outputs=jv,
                    retain_graph=True,
                    allow_unused=True,
                )
                jt_jv = [h if h is not None else torch.zeros_like(p) for h, p in zip(jt_jv, params)]
                flat_jtjv = _flatten_tensors(jt_jv).detach()
                return flat_jtjv + damping * v

        trust_radius = float(group["trust_radius"])
        step_vec, hit_boundary, _ = self._steihaug_cg(
            grad=flat_grad,
            matvec=gn_matvec,
            radius=trust_radius,
            cg_max_iters=int(group["cg_max_iters"]),
            cg_tol=float(group["cg_tol"]),
        )

        Bp = gn_matvec(step_vec)
        predicted_reduction = -(
            torch.dot(flat_grad, step_vec).item() + 0.5 * torch.dot(step_vec, Bp).item()
        )
        if predicted_reduction <= 1e-16:
            group["trust_radius"] = max(1e-8, 0.25 * trust_radius)
            group["damping"] = min(1e3, damping * 2.0)
            return base_loss.detach()

        old_params = [p.detach().clone() for p in params]
        self._apply_step(params, step_vec, scale=1.0)
        with torch.enable_grad():
            trial_loss, _ = closure()
            trial_loss = trial_loss.detach()

        actual_reduction = base_loss.detach().item() - trial_loss.item()
        rho = actual_reduction / predicted_reduction

        if rho < 0.25:
            trust_radius *= 0.25
            group["damping"] = min(1e3, damping * 1.5)
        elif rho > 0.75 and hit_boundary:
            trust_radius = min(2.0 * trust_radius, float(group["max_trust_radius"]))
            group["damping"] = max(1e-6, damping * 0.8)
        group["trust_radius"] = max(1e-8, trust_radius)

        if rho <= float(group["acceptance_eta"]):
            for p, old in zip(params, old_params):
                p.copy_(old)
            group["damping"] = min(1e3, float(group["damping"]) * 1.2)
            return base_loss.detach()
        return trial_loss

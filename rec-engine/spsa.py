# rec-engine/spsa.py

import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Callable, Tuple

class SPSA(Optimizer):
    """
    Implements the Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.
    
    SPSA is a gradient-free optimization method that estimates the gradient by
    evaluating the loss function at two perturbed points. It can be effective for
    escaping local minima in high-dimensional, non-convex optimization problems.

    Args:
        params (Iterable): Iterable of parameters to optimize.
        a (float): SPSA tuning parameter for step size scaling.
        c (float): SPSA tuning parameter for perturbation size.
        A (float): SPSA stability constant.
        alpha (float): SPSA step size decay rate.
        gamma (float): SPSA perturbation decay rate.
    """
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        a: float = 0.01,
        c: float = 0.01,
        A: float = 100.0,
        alpha: float = 0.602,
        gamma: float = 0.101
    ) -> None:
        if not 0.0 <= a: raise ValueError(f"Invalid 'a' value: {a}")
        if not 0.0 <= c: raise ValueError(f"Invalid 'c' value: {c}")

        defaults = dict(a=a, c=c, A=A, alpha=alpha, gamma=gamma)
        super().__init__(params, defaults)
        self.k: int = 0

    def _get_perturbation(self, p: torch.Tensor) -> torch.Tensor:
        """Generates a random perturbation vector via Rademacher distribution."""
        return (torch.bernoulli(torch.ones_like(p) * 0.5) * 2 - 1).to(p.device)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model and returns a tuple
                                where the loss is the first element.
        """
        self.k += 1
        ak = self.param_groups[0]['a'] / (self.k + self.param_groups[0]['A']) ** self.param_groups[0]['alpha']
        ck = self.param_groups[0]['c'] / self.k ** self.param_groups[0]['gamma']

        for group in self.param_groups:
            all_params = [p for p in group['params'] if p.requires_grad]
            if not all_params:
                continue

            flat_params = torch.cat([p.flatten() for p in all_params])
            delta = self._get_perturbation(flat_params)
            
            # Positive perturbation
            offset = 0
            for p in all_params:
                numel = p.numel()
                p.add_(delta[offset : offset + numel].view_as(p), alpha=ck)
                offset += numel
            
            loss_plus, *_ = closure()

            # Negative perturbation
            offset = 0
            for p in all_params:
                numel = p.numel()
                p.add_(delta[offset : offset + numel].view_as(p), alpha=-2 * ck)
                offset += numel

            loss_minus, *_ = closure()
            
            # Restore original parameters
            offset = 0
            for p in all_params:
                numel = p.numel()
                p.add_(delta[offset : offset + numel].view_as(p), alpha=ck)
                offset += numel

            # Estimate gradient and update
            ghat = (loss_plus - loss_minus) / (2 * ck * delta + 1e-8)
            offset = 0
            for p in all_params:
                numel = p.numel()
                p.add_(ghat[offset : offset + numel].view_as(p), alpha=-ak)
                offset += numel
        
        return loss_plus
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ItoDiffusion(ABC):
    """
    Abstract base class for Ito diffusions:
        dX_t = a(t, X_t) dt + b(t, X_t) dW_t
    """

    seed: int | None = field(default=None, kw_only=True)

    @abstractmethod
    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        """a(t, x)"""
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        """b(t, x)"""
        raise NotImplementedError

    def simulate_paths(
        self,
        x0: float,
        t0: float,
        t1: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
    ):
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0.")

        use_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(use_seed)

        dt = (t1 - t0) / n_steps
        sqrt_dt = np.sqrt(dt)

        times = np.linspace(t0, t1, n_steps + 1)
        x = np.full((n_paths,), x0)

        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = x

        dW = np.empty((n_paths, n_steps), dtype=float)

        for k in range(n_steps):
            t = times[k]

            Z = rng.standard_normal(n_paths)
            dw = sqrt_dt * Z

            a = self.drift(t, x)
            b = self.diffusion(t, x)

            if a.shape != x.shape or b.shape != x.shape:
                raise ValueError(
                    "drift/diffusion must return arrays of shape (n_paths,). "
                    f"Got drift {a.shape}, diffusion {b.shape}, expected {x.shape}."
                )

            x = x + a * dt + b * dw
            paths[:, k + 1] = x

            dW[:, k] = dw

        return paths, times, dW


@dataclass(frozen=True)
class GBM(ItoDiffusion):
    """
    Geometric Brownian Motion:
        dS_t = mu * S_t dt + sigma * S_t dW_t
    """

    mu: float
    sigma: float

    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.mu * x

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.sigma * x


@dataclass(frozen=True)
class HullWhite(ItoDiffusion):
    """
    1-factor Hull–White (Ornstein–Uhlenbeck) short rate model:
        dr_t = a(θ(t) - r_t) dt + σ dW_t
    """

    a: float
    sigma: float
    theta: Callable[[float], float]

    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.a * (self.theta(t) - x)

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, self.sigma)

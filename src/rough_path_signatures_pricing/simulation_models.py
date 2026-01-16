from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class Simulator(ABC):
    """
    Abstract base class for path simulators.
    """

    x0: float
    t0: float
    t1: float
    n_steps: int
    seed: int | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.t1 <= self.t0:
            raise ValueError("t1 must be greater than t0.")

    @abstractmethod
    def simulate_paths(self, n_paths: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths. Returns (paths, times, dW) or similar structure depending on implementation.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ItoDiffusion(Simulator):
    """
    Abstract base class for Ito diffusions:
        dX_t = a(t, X_t) dt + b(t, X_t) dW_t
    """

    @abstractmethod
    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        """a(t, x)"""
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        """b(t, x)"""
        raise NotImplementedError

    def simulate_paths(self, n_paths: int):
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")

        rng = np.random.default_rng(self.seed)

        dt = (self.t1 - self.t0) / self.n_steps
        sqrt_dt = np.sqrt(dt)

        times = np.linspace(self.t0, self.t1, self.n_steps + 1)
        x = np.full((n_paths,), self.x0, dtype=float)

        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = x

        dW = np.empty((n_paths, self.n_steps), dtype=float)

        for k in range(self.n_steps):
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

    def simulate_paths(self, n_paths: int):
        """
        Simulate GBM using the exact discrete-time solution (multiplicative log update):

        S_{t+dt} = S_t * exp((mu - 0.5*sigma^2) dt + sigma * dW)
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")

        rng = np.random.default_rng(self.seed)

        dt = (self.t1 - self.t0) / self.n_steps

        times = np.linspace(self.t0, self.t1, self.n_steps + 1)
        x = np.full((n_paths,), self.x0, dtype=float)

        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = x

        dW = np.empty((n_paths, self.n_steps), dtype=float)

        for k in range(self.n_steps):
            Z = rng.standard_normal(n_paths)
            dw = np.sqrt(dt) * Z

            # Exact solution step: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)dt + sigma*dW)
            factor = np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dw)
            x = x * factor

            paths[:, k + 1] = x
            dW[:, k] = dw

        return paths, times, dW


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


@dataclass(frozen=True)
class GARCH(Simulator):
    """
    GARCH(1,1) model for discrete-time price simulation.
    """

    omega: float
    alpha: float
    beta: float
    mu: float

    def simulate_paths(self, n_paths: int):
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")

        rng = np.random.default_rng(self.seed)
        dt = (self.t1 - self.t0) / self.n_steps
        sqrt_dt = np.sqrt(dt)

        times = np.linspace(self.t0, self.t1, self.n_steps + 1)
        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = self.x0

        # init the variance at long-term average
        v = np.full(n_paths, self.omega / (1 - self.alpha - self.beta))

        # Store effective dW for consistency (normalized innovations)
        dW = np.empty((n_paths, self.n_steps), dtype=float)

        for k in range(self.n_steps):
            z = rng.standard_normal(n_paths)
            dw = sqrt_dt * z

            # S_{t+1} = S_t * exp((mu - 0.5*v)*dt + sqrt(v)*dw)
            paths[:, k + 1] = paths[:, k] * np.exp(
                (self.mu - 0.5 * v) * dt + np.sqrt(v) * dw
            )

            # Variance update: v_{t+1} = omega + alpha*(z^2*v_t) + beta*v_t
            # Note: GARCH typically defined on returns epsilon = z * sqrt(v)
            # Here we follow a standard GARCH(1,1) process ensuring positivity
            v = self.omega + self.alpha * (z**2 * v) + self.beta * v

            dW[:, k] = dw

        return paths, times, dW


@dataclass(frozen=True)
class JumpDiffusion(ItoDiffusion):
    """
    Merton Jump Diffusion: dS_t = mu*S_t*dt + sigma*S_t*dW_t + S_t*dJ_t
    """

    mu: float
    sigma: float
    lambda_j: float  # intensity of jumps
    mu_j: float  # mean jump size
    sigma_j: float  # jump volatility

    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.mu * x

    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.sigma * x

    def simulate_paths(self, n_paths: int):
        rng = np.random.default_rng(self.seed)

        # Use parent simulation for the diffusion part (GBM)
        # Note: Ideally we would reuse super().simulate_paths logic but we need to inject jumps.
        # Calling super().simulate_paths(n_paths) gives us paths, times, dW.
        # But applying jumps post-hoc to a multiplicative process is tricky if paths are already built.
        # It's better to reimplement the loop or composed it properly.
        # Here we reimplement to ensure correct interleaving of jumps.

        dt = (self.t1 - self.t0) / self.n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(self.t0, self.t1, self.n_steps + 1)

        x = np.full((n_paths,), self.x0, dtype=float)
        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = x

        dW = np.empty((n_paths, self.n_steps), dtype=float)

        for k in range(self.n_steps):
            # 1. Diffusion Step (exact GBM update)
            Z = rng.standard_normal(n_paths)
            dw = sqrt_dt * Z

            # S_diff = S * exp(...)
            factor_diff = np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dw)

            # 2. Jump Step
            # Number of jumps in this interval ~ Poisson(lambda * dt)
            # For small dt, mostly 0 or 1.
            n_jumps = rng.poisson(self.lambda_j * dt, size=n_paths)

            # Total jump multiplier
            # If N jumps, multiplier is product of (1+J_i) -> exp(sum log(1+J_i))
            # Merton Jump magnitude Y = log(1+k) ~ N(mu_j, sigma_j^2) implies jump factor is lognormal
            # log_jump ~ Normal(mu_j, sigma_j)

            jump_factor = np.ones(n_paths)
            if np.any(n_jumps > 0):
                # Efficiently handle multiple jumps (though rare in small dt)
                # Sum of normals is normal.
                # Total log-jump for path i is Sum_{j=1}^{N_i} Y_{ij}
                # Distribution of sum: Mean = N_i * mu_j, Var = N_i * sigma_j^2

                # Identify paths with jumps
                has_jumps = n_jumps > 0
                counts = n_jumps[has_jumps]

                # Sample total log-jump magnitude directly
                total_mean = counts * self.mu_j
                total_std = np.sqrt(counts) * self.sigma_j
                total_log_jump = rng.normal(total_mean, total_std)

                jump_factor[has_jumps] = np.exp(total_log_jump)

            x = x * factor_diff * jump_factor

            paths[:, k + 1] = x
            dW[:, k] = dw

        return paths, times, dW


@dataclass(frozen=True)
class RoughVolatility(Simulator):
    """
    Simplified Rough Fractional Stochastic Volatility (RFSV) model.
    """

    H: float  # Hurst exponent (H < 0.5 for rough vol)
    eta: float  # vol of vol

    def simulate_paths(self, n_paths: int):
        """Approximates rough vol using a fractional Brownian motion kernel."""
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")

        rng = np.random.default_rng(self.seed)
        dt = (self.t1 - self.t0) / self.n_steps
        times = np.linspace(self.t0, self.t1, self.n_steps + 1)

        paths = np.empty((n_paths, self.n_steps + 1), dtype=float)
        paths[:, 0] = self.x0

        # Volatility process
        vol = np.zeros((n_paths, self.n_steps + 1))
        vol[:, 0] = (
            0.2  # Initial vol hardcoded in original, should probably be parameter
        )

        dW = np.empty((n_paths, self.n_steps), dtype=float)

        # Simplified RFSV simulation
        # Note: Proper simulation requires Cholesky decomposition of covariance matrix
        # for fractional Brownian motion, or hybrid scheme.
        # This loop-based approximation from the original code is very rough but fits structure.

        for k in range(self.n_steps):
            Z = rng.standard_normal(n_paths)
            dw = np.sqrt(dt) * Z

            # Naive fractional implementation (needs correct kernel integration usually)
            # kept from original user code structure but cleaned up inputs
            vol_inc = self.eta * ((k + 1) * dt) ** (self.H - 0.5) * dw
            vol[:, k + 1] = np.abs(vol[:, k] + vol_inc)

            # Price update
            paths[:, k + 1] = paths[:, k] * (1 + vol[:, k] * dw)

            dW[:, k] = dw

        return paths, times, dW

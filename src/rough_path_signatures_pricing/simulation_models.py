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

@dataclass(frozen=True)
class GARCH(ItoDiffusion):
    """
    GARCH(1,1) model for discrete-time price simulation.
    Note: Standard drift/diffusion methods are bypassed for discrete update logic.
    """
    omega: float
    alpha: float
    beta: float
    mu: float

    def drift(self, t, x): return np.zeros_like(x) # useless since discrete updates
    def diffusion(self, t, x): return np.zeros_like(x)

    def simulate_paths(self, x0, t0, t1, n_steps, n_paths, seed=None):
        rng = np.random.default_rng(seed)
        dt = (t1 - t0) / n_steps
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0
        
        # init the variance at long-term average
        v = np.full(n_paths, self.omega / (1 - self.alpha - self.beta))
        
        for k in range(n_steps):
            z = rng.standard_normal(n_paths)
            # update the price with garch formula: S_{t+1} = S_t * exp((mu - 0.5*v)*dt + sqrt(v*dt)*z)
            paths[:, k+1] = paths[:, k] * np.exp((self.mu - 0.5 * v) * dt + np.sqrt(v * dt) * z)
            # Variance update: v_{t+1} = omega + alpha*(z^2*v_t) + beta*v_t
            v = self.omega + self.alpha * (z**2 * v * dt) + self.beta * v
            
        return np.linspace(t0, t1, n_steps + 1), paths

@dataclass(frozen=True)
class JumpDiffusion(ItoDiffusion):
    """
    Merton Jump Diffusion: dS_t = mu*S_t*dt + sigma*S_t*dW_t + S_t*dJ_t
    """
    mu: float
    sigma: float
    lambda_j: float  # intensity of jumps
    mu_j: float      # mean jump size
    sigma_j: float   # jump volatility

    def drift(self, t, x): return self.mu * x
    def diffusion(self, t, x): return self.sigma * x

    def simulate_paths(self, x0, t0, t1, n_steps, n_paths, seed=None):
        rng = np.random.default_rng(seed)
        dt = (t1 - t0) / n_steps
        times, paths = super().simulate_paths(x0, t0, t1, n_steps, n_paths, seed)
        
        # adding jumps wrt a Poisson process
        n_jumps = rng.poisson(self.lambda_j * dt, size=(n_paths, n_steps))
        for k in range(n_steps):
            if np.any(n_jumps[:, k] > 0):
                # magnitude of the jump is J = exp(mu_j + sigma_j * Z) - 1 for Z normal rv
                jump_magnitudes = np.exp(rng.normal(self.mu_j, self.sigma_j, n_paths)) - 1
                paths[:, k+1] += paths[:, k] * (n_jumps[:, k] * jump_magnitudes)
                
        return times, paths

@dataclass(frozen=True)
class RoughVolatility(ItoDiffusion):
    """
    Simplified Rough Fractional Stochastic Volatility (RFSV) model.
    Uses the Hybrid Scheme or a Power-law kernel approximation.
    """
    H: float        # Hurst exponent (H < 0.5 for rough vol)
    eta: float      # vol of vol
    lambda_reg: float # Mean reversion speed
    
    def drift(self, t, x): return np.zeros_like(x)
    def diffusion(self, t, x): return np.zeros_like(x)

    # redefine simulate_paths to implement rough vol logic since non-Markovian
    def simulate_paths(self, x0, t0, t1, n_steps, n_paths, seed=None):
        """Approximates rough vol using a fractional Brownian motion kernel."""
        rng = np.random.default_rng(seed)
        dt = (t1 - t0) / n_steps
        times = np.linspace(t0, t1, n_steps + 1)
        
        # here we simplify : generate fractional noise kernel
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0
        
        # vol process follows a fractional kernel integration
        vol = np.zeros((n_paths, n_steps + 1))
        vol[:, 0] = 0.2 # initial vol
        
        # this is a very simplified rough kernel update
        # apparently actual implementation usually requires FFT or Cholesky?
        for k in range(n_steps):
            dw = rng.standard_normal(n_paths) * np.sqrt(dt)
            vol_inc = self.eta * (k*dt)**(self.H - 0.5) * dw
            vol[:, k+1] = np.abs(vol[:, k] + vol_inc) 
            paths[:, k+1] = paths[:, k] * (1 + 0.05*dt + vol[:, k]*rng.standard_normal(n_paths)*np.sqrt(dt))
            
        return times, paths
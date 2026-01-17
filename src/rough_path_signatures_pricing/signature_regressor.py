from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from esig import stream2sig

from .simulation_models import Simulator


@dataclass
class SignaturePricer:
    simulator: Simulator
    func: Callable[[np.ndarray, float | np.ndarray], np.ndarray]
    signature_degree: int
    paths_: np.ndarray = field(init=False, default=None)
    times_: np.ndarray = field(init=False, default=None)
    signatures_: np.ndarray = field(init=False, default=None)
    lam_: float = field(init=False, default=None)
    implied_expected_signature_: np.ndarray = field(init=False, default=None)
    coeffs_: np.ndarray = field(init=False, default=None)

    def _get_signatures(self, paths: np.ndarray, t: np.ndarray) -> np.ndarray:
        sig_list = []

        P, L = paths.shape

        for i in range(P):
            path = paths[i]

            t_lag = np.empty_like(t)
            t_lag[0] = t[0]
            t_lag[1:] = t[:-1]

            path_lag = np.empty_like(path)
            path_lag[0] = path[0]
            path_lag[1:] = path[:-1]

            path = np.column_stack([t_lag, path_lag, path])
            sig_list.append(stream2sig(path, self.signature_degree))

        signatures = np.vstack(sig_list)

        return signatures

    def _get_linear_functional(
        self, signatures: np.ndarray, payoff: np.ndarray, lam: float
    ) -> np.ndarray:
        return np.linalg.solve(
            signatures.T @ signatures + lam * np.eye(signatures.shape[1]),
            signatures.T @ payoff,
        )

    def _get_implied_expected_signature(self, y: np.ndarray, lam: float) -> np.ndarray:
        return np.linalg.solve(
            self.coeffs_.T @ self.coeffs_ + lam * np.eye(self.coeffs_.shape[1]),
            self.coeffs_.T @ y,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_paths: int = 10000,
        lam: float = 0,
    ) -> None:
        self.paths_, self.times_, _ = self.simulator.simulate_paths(n_paths)
        self.signatures_ = self._get_signatures(self.paths_, self.times_)
        self.lam_ = lam

        coeffs = []
        for param in X:
            param_payoffs = self.func(self.paths_, param)
            param_coeff = self._get_linear_functional(
                self.signatures_, param_payoffs, lam
            )
            coeffs.append(param_coeff)

        self.coeffs_ = np.vstack(coeffs)

        self.implied_expected_signature_ = self._get_implied_expected_signature(y, lam)

    def predict(self, X: float | np.ndarray) -> np.ndarray:
        if type(X) is not np.ndarray:
            X = np.array([X])

        coeffs = []
        for param in X:
            param_payoffs = self.func(self.paths_, param)
            param_coeff = self._get_linear_functional(
                self.signatures_, param_payoffs, self.lam_
            )
            coeffs.append(param_coeff)
        coeffs_stacked = np.vstack(coeffs)

        return coeffs_stacked @ self.implied_expected_signature_

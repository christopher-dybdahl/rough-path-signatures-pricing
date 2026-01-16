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
        self, signatures: np.ndarray, payoff: np.ndarray
    ) -> np.ndarray:
        return np.linalg.inv(signatures.T @ signatures) @ signatures.T @ payoff

    def _get_implied_expected_signatures(self, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.coeffs_.T @ self.coeffs_) @ self.coeffs_.T @ y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_paths: int = 10000,
    ) -> np.ndarray:
        self.paths, self.times, _ = self.simulator.simulate_paths(n_paths)
        self.signatures_ = self._get_signatures(self.paths, self.times)

        coeffs = []
        for param in X:
            param_payoffs = self.func(self.paths, param)
            param_coeff = self._get_linear_functional(self.signatures_, param_payoffs)
            coeffs.append(param_coeff)

        self.coeffs_ = np.vstack(coeffs)

        self.implied_expected_signature_ = self._get_implied_expected_signatures(y)

    def predict(self, X: float | np.ndarray) -> np.ndarray:
        if type(X) is not np.ndarray:
            X = np.array([X])

        coeffs = []
        for param in X:
            param_payoffs = self.func(self.paths, param)
            param_coeff = self._get_linear_functional(self.signatures_, param_payoffs)
            coeffs.append(param_coeff)
        coeffs_stacked = np.vstack(coeffs)

        return coeffs_stacked @ self.implied_expected_signature_

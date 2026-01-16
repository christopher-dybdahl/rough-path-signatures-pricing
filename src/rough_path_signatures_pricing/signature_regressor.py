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
    coeffs_: np.ndarray = field(init=False, default=None)
    implied_expected_signature_: np.ndarray = field(init=False, default=None)

    def _get_signatures(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        sig_list = []

        P, L = X.shape

        for i in range(P):
            path = X[i]

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

    def _get_linear_functional(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(X.T @ X) @ X.T @ y

    def _get_implied_expected_signatures(self, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.coeffs_.T @ self.coeffs_) @ self.coeffs_.T @ y

    def fit(
        self,
        y: np.ndarray,
        params: np.ndarray,
        n_paths: int = 10000,
    ) -> np.ndarray:
        paths, times, _ = self.simulator.simulate_paths(n_paths)

        signatures = self._get_signatures(paths, times)

        coeffs = []
        for param in params:
            param_payoffs = self.func(paths, param)
            param_coeff = self._get_linear_functional(signatures, param_payoffs)
            coeffs.append(param_coeff)

        self.coeffs_ = np.vstack(coeffs)

        self.implied_expected_signature_ = self._get_implied_expected_signatures(y)

    def predict(self, param: float, n_paths: int = 10000) -> np.ndarray:
        paths, times, _ = self.simulator.simulate_paths(n_paths)

        signatures = self._get_signatures(paths, times)
        param_payoffs = self.func(paths, param)

        param_coeff = self._get_linear_functional(signatures, param_payoffs)

        return self.implied_expected_signature_ @ param_coeff

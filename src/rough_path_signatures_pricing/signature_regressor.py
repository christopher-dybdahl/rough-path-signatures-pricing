from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from esig import stream2sig


@dataclass
class SignaturePricer:
    signature_degree: int
    payoff: Callable[[np.ndarray], np.ndarray]
    coeffs_: np.ndarray = field(init=False, default=None)

    def _get_signatures(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        sig_list = []

        P, L = X.shape

        for i in range(P):
            price = X[i]

            t_lag = np.empty_like(t)
            t_lag[0] = t[0]
            t_lag[1:] = t[:-1]

            p_lag = np.empty_like(price)
            p_lag[0] = price[0]
            p_lag[1:] = price[:-1]

            path = np.column_stack([t_lag, p_lag, price])
            sig_list.append(stream2sig(path, self.signature_degree))

        signatures = np.vstack(sig_list)

        return signatures

    def fit(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        payoffs = self.payoff(X)
        signatures = self._get_signatures(X, t)

        self.coeffs_ = np.linalg.inv(signatures.T @ signatures) @ signatures.T @ payoffs

    def _get_implied_expected_signatures(self, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.coeffs_.T @ self.coeffs_) @ self.coeffs_.T @ y

    def price(self, y: np.ndarray) -> np.ndarray:
        implied_signatures = self._get_implied_expected_signatures(y)
        return float(implied_signatures @ self.coeffs_)

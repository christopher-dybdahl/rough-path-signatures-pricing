from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from esig import stream2sig


@dataclass
class SignaturePricer:
    signature_degree: int
    payoff: Callable[[np.ndarray], np.ndarray]
    coeffs_: np.ndarray = field(init=False, default=None)

    def _get_signatures(self, X: np.ndarray) -> np.ndarray:
        return stream2sig(X, self.signature_degree)

    def fit(self, X: np.ndarray) -> np.ndarray:
        payoffs = self.payoff(X)
        signatures = self._get_signatures(X)

        self.coeffs_ = np.linalg.inv(payoffs.T @ payoffs) @ payoffs.T @ signatures

    def _get_implied_signatures(self, y: np.ndarray) -> np.ndarray:
        return np.linalg.inv(self.coeffs_.T @ self.coeffs_) @ self.coeffs_.T @ y

    def price(self, y: np.ndarray) -> np.ndarray:
        implied_signatures = self._get_implied_signatures(self, y)
        return float(implied_signatures @ self.coeffs_)

# Rough Path Signatures Pricing

This repository implements the pricing model from Lyons et al. (2019).

## Setup and Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Prerequisites**: Ensure you have Python 3.12+ and Poetry installed.

2.  **Install Dependencies**:
    ```bash
    poetry install
    ```
    Poetry installs the project in editable mode by default.

### Alternative Installation (pip / Conda)

A `requirements.txt` file is also provided for environments where Poetry is not used (e.g., Conda).

```bash
pip install -r requirements.txt
pip install -e .  # Install the project in editable mode
```

## Usage

To use the pricing package, follow these five steps:

1.  **Initialize a Simulator**: Choose a stochastic process (e.g., GBM, Hull-White) to generate market paths.
2.  **Define the Payoff Function**: Implement the payoff logic (e.g., European Call) accepting paths and a parameter (strike).
3.  **Initialize the Pricer**: Create a `SignaturePricer` instance with the simulator and payoff function.
4.  **Fit to Market Data**: Calibrate the model using observed option prices and their corresponding parameters (e.g. strikes).
5.  **Predict**: Price new options by supplying their parameters.

### Example

```python
import numpy as np
from rough_path_signatures_pricing import simulation_models as sm
from rough_path_signatures_pricing import signature_regressor as sr

# 1. Create a generator object (Simulator)
# Example: Geometric Brownian Motion parameters
gbm_params = {
    "x0": 100.0, "t0": 0, "t1": 1, "n_steps": 252,
    "mu": 0.05, "sigma": 0.2, "seed": 42
}
simulator = sm.GBM(**gbm_params)

# 2. Define the option payoff function
# SignaturePricer expects: func(paths, parameter) -> payoffs
def european_call_payoff(paths: np.ndarray, strike: float) -> np.ndarray:
    final_prices = paths[:, -1]
    return np.maximum(final_prices - strike, 0.0)

# 3. Create the pricing object
pricer = sr.SignaturePricer(
    simulator=simulator,
    func=european_call_payoff,
    signature_degree=5  # Degree of signature truncation
)

# 4. Fit to market data (Calibration)
# X: Option parameters (e.g., Strikes)
# y: Market Prices
market_strikes = np.array([90, 100, 110])
market_prices = np.array([15.2, 8.5, 3.1])  # Dummy market data

# n_paths: Number of simulation paths for calibration
# lam: Regularization parameter (Ridge regression)
pricer.fit(X=market_strikes, y=market_prices, n_paths=10000)

# 5. Predict prices for unobserved parameters
target_strikes = np.array([95, 105])
predicted_prices = pricer.predict(X=target_strikes)

print(f"Predicted Prices for strikes {target_strikes}: {predicted_prices}")
```

## References

* Lyons, T., Nejad, S., & Perez Arribas, I. (2019). Numerical Method for Model-free Pricing of Exotic Derivatives in Discrete Time Using Rough Path Signatures. Applied Mathematical Finance, 26(6), 583–597. https://doi.org/10.1080/1350486X.2020.1726784
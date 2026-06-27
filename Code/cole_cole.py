"""
Cole-Cole bioimpedance model fitting.

Model: Z(ω) = R_inf + (R0 - R_inf) / (1 + (jωτ)^α)

Parameters:
    R0    - resistance at DC (low-freq limit, reflects extracellular fluid volume)
    R_inf - resistance at infinite frequency (total water)
    fc    - characteristic frequency [Hz], τ = 1/(2π·fc)
    alpha - dispersion exponent (0 < α ≤ 1)
"""
import numpy as np
from scipy.optimize import least_squares


def cole_cole_complex(freqs: np.ndarray, R0: float, R_inf: float, fc: float, alpha: float) -> np.ndarray:
    """Return complex impedance array for given frequencies."""
    omega = 2 * np.pi * np.asarray(freqs)
    tau = 1.0 / (2 * np.pi * fc)
    term = (1j * omega * tau) ** alpha
    return R_inf + (R0 - R_inf) / (1.0 + term)


def _residuals(params, freqs, z_real, z_imag):
    R0, R_inf, log_fc, alpha = params
    fc = np.exp(log_fc)
    z = cole_cole_complex(freqs, R0, R_inf, fc, alpha)
    # normalise by |Z| magnitude to weight frequencies equally
    scale = np.sqrt(z_real**2 + z_imag**2)
    res_real = (z.real - z_real) / scale
    res_imag = (z.imag - z_imag) / scale
    return np.concatenate([res_real, res_imag])


def fit_cole_cole(freqs: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray) -> dict:
    """
    Fit Cole-Cole parameters to a measured impedance spectrum.
    Returns dict with R0, R_inf, fc, alpha, and fit_rmse.
    Returns NaNs if fitting fails.
    """
    freqs = np.asarray(freqs, dtype=float)
    z_real = np.asarray(z_real, dtype=float)
    z_imag = np.asarray(z_imag, dtype=float)

    # initial guesses from data
    R_inf_0 = z_real[-1]          # high-freq real part ≈ R_inf
    R0_0    = z_real[0]           # low-freq real part ≈ R0
    # characteristic frequency: where |imag| is maximum
    fc_0    = freqs[np.argmin(z_imag)]  # imag is negative, min = most negative
    alpha_0 = 0.8

    x0 = [R0_0, R_inf_0, np.log(fc_0), alpha_0]
    bounds = (
        [R_inf_0 * 0.5, R_inf_0 * 0.1, np.log(freqs[0]), 0.1],
        [R0_0    * 2.0, R0_0    * 2.0, np.log(freqs[-1]), 1.0],
    )

    try:
        result = least_squares(
            _residuals, x0, bounds=bounds,
            args=(freqs, z_real, z_imag),
            max_nfev=1000, ftol=1e-8, xtol=1e-8,
        )
        R0, R_inf, log_fc, alpha = result.x
        fc = np.exp(log_fc)
        z_fit = cole_cole_complex(freqs, R0, R_inf, fc, alpha)
        rmse = np.sqrt(np.mean((z_fit.real - z_real)**2 + (z_fit.imag - z_imag)**2))
        return {"R0": R0, "R_inf": R_inf, "fc": fc, "alpha": alpha, "fit_rmse": rmse}
    except Exception:
        return {"R0": np.nan, "R_inf": np.nan, "fc": np.nan, "alpha": np.nan, "fit_rmse": np.nan}

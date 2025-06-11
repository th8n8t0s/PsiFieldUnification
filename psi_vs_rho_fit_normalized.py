import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Domain for radial coordinate
r = np.linspace(0.01, 10, 500)

# Experimental reference profile (normalized Gaussian-like proton charge density)
def experimental_rho(r):
    sigma = 0.8
    norm_factor = 1 / (sigma * np.sqrt(2 * np.pi))
    return norm_factor * np.exp(-r**2 / (2 * sigma**2))

# Ψ field ansatz: tunable Gaussian + offset
def predicted_psi(r, a, b):
    return np.exp(-a * r**2) + b

# Compute spherical Laplacian of Ψ
def laplacian_psi(r, a, b):
    psi = predicted_psi(r, a, b)
    dpsi = np.gradient(psi, r)
    d2psi = np.gradient(dpsi, r)
    return d2psi + 2/r * dpsi

# Fit function: scale × curvature to match ρ(r)
def fit_func(r, a, b, scale):
    return scale * laplacian_psi(r, a, b)

# Generate target profile
rho_exp = experimental_rho(r)

# Fit curvature model to target density
params, _ = curve_fit(fit_func, r, rho_exp, p0=[0.1, 0.0, 1.0])
a_fit, b_fit, scale_fit = params

# Evaluate best-fit Ψ and Laplacian
psi_fit = predicted_psi(r, a_fit, b_fit)
lap_psi_fit = laplacian_psi(r, a_fit, b_fit)
scaled_lap_psi_fit = scale_fit * lap_psi_fit

# Normalize all to max = 1 for visual comparability
rho_exp_norm = rho_exp / np.max(rho_exp)
psi_fit_norm = psi_fit / np.max(psi_fit)
lap_psi_norm = scaled_lap_psi_fit / np.max(scaled_lap_psi_fit)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(r, rho_exp_norm, label='Normalized Experimental ρ(r)', linewidth=2)
plt.plot(r, lap_psi_norm, '--', label='Normalized Scaled ∇²Ψ(r)', linewidth=2)
plt.plot(r, psi_fit_norm, ':', label='Normalized Fitted Ψ(r)', linewidth=2)
plt.xlabel("Radial Distance r", fontsize=12)
plt.ylabel("Normalized Amplitude", fontsize=12)
plt.title("Ψ-Field Derived Curvature vs Experimental Density", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/imman/source/repos/psi_vs_rho_fit_normalized.png")
plt.show()


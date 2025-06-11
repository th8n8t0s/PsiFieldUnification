import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Simulated Ψ profile from previous assumptions
r = np.linspace(0.01, 10, 500)

# Experimental-like profile (proton charge density fit)
def experimental_rho(r):
    return np.exp(-r**2 / (2 * 0.8**2)) / (0.8 * np.sqrt(2 * np.pi))

# Predicted Ψ(r) form from curvature equation
def predicted_psi(r, a, b):
    return np.exp(-a * r**2) + b

# Generate synthetic experimental data
rho_exp = experimental_rho(r)

# Fit predicted Ψ to match experimental ρ via second derivative approximation
# Laplacian in spherical symmetry: ∇²Ψ ≈ (d²Ψ/dr² + 2/r dΨ/dr)
def laplacian_psi(r, a, b):
    psi = predicted_psi(r, a, b)
    dpsi = np.gradient(psi, r)
    d2psi = np.gradient(dpsi, r)
    return d2psi + 2/r * dpsi

# Use curve fitting to find best (a, b) to match ∇²Ψ ~ ρ_exp
def fit_func(r, a, b, scale):
    return scale * laplacian_psi(r, a, b)

params, _ = curve_fit(fit_func, r, rho_exp, p0=[0.1, 0.0, 1.0])
a_fit, b_fit, scale_fit = params

# Compute fitted Psi and its curvature
psi_fit = predicted_psi(r, a_fit, b_fit)
lap_psi_fit = laplacian_psi(r, a_fit, b_fit)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(r, rho_exp, label='Experimental ρ(r)', linewidth=2)
plt.plot(r, scale_fit * lap_psi_fit, label='Scaled ∇²Ψ(r)', linestyle='--')
plt.plot(r, psi_fit, label='Fitted Ψ(r)', linestyle=':')
plt.xlabel("r")
plt.ylabel("Density / Curvature")
plt.title("Quantitative Comparison: Ψ-derived Curvature vs Experimental ρ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/imman/source/repos/psi_vs_rho_fit.png")

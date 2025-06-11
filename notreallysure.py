import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Parameters
L = 100
T = 200
dx = 1.0
dt = 0.1
x = np.linspace(0, L, int(L/dx))
t = np.linspace(0, T, int(T/dt))
Nx = len(x)
Nt = len(t)

# Define several Psi profiles
def psi_flat(x):  # No curvature
    return np.ones_like(x)

def psi_gaussian(x):  # Single well
    return 1.0 - 0.8 * np.exp(-0.01 * (x - L/2)**2)

def psi_double_well(x):  # Two wells
    return 1.0 - 0.5 * np.exp(-0.01 * (x - L/3)**2) - 0.5 * np.exp(-0.01 * (x - 2*L/3)**2)

psi_profiles = {
    "Flat": psi_flat(x),
    "Single Well": psi_gaussian(x),
    "Double Well": psi_double_well(x)
}

entropy_results = {}

# Evolve for each Psi profile
for label, psi in psi_profiles.items():
    # Initialize entangled fields φ1 and φ2
    phi1 = np.zeros((Nt, Nx))
    phi2 = np.zeros((Nt, Nx))
    
    # Initial condition: symmetric Gaussian pulses
    phi1[0] = np.exp(-0.1 * (x - L/3)**2)
    phi2[0] = np.exp(-0.1 * (x - 2*L/3)**2)
    
    # First time step
    phi1[1] = phi1[0]
    phi2[1] = phi2[0]
    
    # Time evolution
    for n in range(1, Nt-1):
        for i in range(1, Nx-1):
            phi1[n+1, i] = (2 * phi1[n, i] - phi1[n-1, i] +
                            psi[i] * dt**2 / dx**2 * (phi1[n, i+1] - 2*phi1[n, i] + phi1[n, i-1]))
            phi2[n+1, i] = (2 * phi2[n, i] - phi2[n-1, i] +
                            psi[i] * dt**2 / dx**2 * (phi2[n, i+1] - 2*phi2[n, i] + phi2[n, i-1]))
    
    # Compute entropy of overlap
    phi_product = phi1 * phi2
    overlap_prob = np.abs(phi_product)**2
    overlap_prob /= np.sum(overlap_prob, axis=1, keepdims=True) + 1e-12  # Normalize across x
    entropy_t = np.array([entropy(p) for p in overlap_prob])
    
    entropy_results[label] = entropy_t

# Plot entropy over time for each profile
plt.figure(figsize=(10, 6))
for label, ent_vals in entropy_results.items():
    plt.plot(t, ent_vals, label=label)

plt.xlabel("Time")
plt.ylabel("Entropy of Field Overlap")
plt.title("Decoherence Metric Across Ψ Profiles")
plt.legend()
plt.grid(True)
plt.tight_layout()

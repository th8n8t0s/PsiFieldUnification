# Unified Lagrangian for Scalar, Vector, and Spinor Fields in a Psi-Curved Background

from sympy import symbols, Function, Derivative, sin

# Coordinates
x, t, r, theta, phi = symbols('x t r theta phi')

# Background coherence field
Psi = Function('Psi')(r)

# --- 1. Scalar Field ---
phi = Function('phi')(t, r)
L_scalar = 0.5 * Psi * (Derivative(phi, t)**2 - Derivative(phi, r)**2) - 0.5 * Psi * phi**2

# --- 2. Vector Field (e.g., Electromagnetic-like, A_mu = [A_t, A_r]) ---
A_t = Function('A_t')(t, r)
A_r = Function('A_r')(t, r)
F_tr = Derivative(A_r, t) - Derivative(A_t, r)
L_vector = -0.25 * Psi**2 * F_tr**2

# --- 3. Spinor Field (simplified 1+1D Dirac form) ---
psi1 = Function('psi1')(t, r)  # upper component
psi2 = Function('psi2')(t, r)  # lower component

L_spinor = Psi * (psi1 * Derivative(psi2, t) - psi2 * Derivative(psi1, t)
                  - psi1 * Derivative(psi2, r) + psi2 * Derivative(psi1, r))

# --- Total Unified Lagrangian ---
L_total = L_scalar + L_vector + L_spinor

print("Unified Lagrangian Density:")
print(L_total)

# Notes:
# - This Lagrangian is symbolic and simplified for 1+1D.
# - Psi acts as both a curvature-modifier and coupling term.
# - Each term shows how Psi modifies energy propagation for that field.
# - Full 3+1D generalization would require tensor field notation and covariant derivatives.


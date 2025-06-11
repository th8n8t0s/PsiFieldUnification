# Tensor Derivation for Psi-Based Metric in 4D Spacetime

import sympy as sp
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, tensorhead

# Define coordinates
coords = sp.symbols('t r theta phi')
t, r, theta, phi = coords

# Define Psi as a function of r
Psi = sp.Function('Psi')(r)

# Define metric components g_{\mu\nu} in terms of Psi (spherical symmetry assumed)
g = sp.MutableDenseNDimArray([[0]*4 for _ in range(4)], (4, 4))

g[0,0] = -Psi

g[1,1] = 1 / Psi

g[2,2] = r**2

g[3,3] = r**2 * sp.sin(theta)**2

# Define inverse metric
g_inv = sp.MutableDenseNDimArray([[0]*4 for _ in range(4)], (4, 4))
g_inv[0,0] = -1 / Psi
g_inv[1,1] = Psi
g_inv[2,2] = 1 / r**2
g_inv[3,3] = 1 / (r**2 * sp.sin(theta)**2)

# Define Christoffel symbols
Gamma = sp.MutableDenseNDimArray([[[0]*4 for _ in range(4)] for _ in range(4)], (4,4,4))

for lam in range(4):
    for mu in range(4):
        for nu in range(4):
            s = 0
            for sigma in range(4):
                dg = sp.diff(g[nu,sigma], coords[mu]) + sp.diff(g[mu,sigma], coords[nu]) - sp.diff(g[mu,nu], coords[sigma])
                s += g_inv[lam,sigma] * dg
            Gamma[lam,mu,nu] = sp.simplify(0.5 * s)

# Ricci tensor
R = sp.MutableDenseNDimArray([[0]*4 for _ in range(4)], (4,4))

for mu in range(4):
    for nu in range(4):
        term = 0
        for lam in range(4):
            term += sp.diff(Gamma[lam,mu,nu], coords[lam])
            for sigma in range(4):
                term += Gamma[lam,mu,sigma] * Gamma[sigma,lam,nu] - Gamma[lam,sigma,nu] * Gamma[sigma,mu,lam]
        R[mu,nu] = sp.simplify(term)

# Ricci scalar
R_scalar = sum([g_inv[mu,nu] * R[mu,nu] for mu in range(4) for nu in range(4)])

# Einstein tensor
G = sp.MutableDenseNDimArray([[0]*4 for _ in range(4)], (4,4))

for mu in range(4):
    for nu in range(4):
        G[mu,nu] = sp.simplify(R[mu,nu] - 0.5 * g[mu,nu] * R_scalar)

# Display Einstein tensor
for mu in range(4):
    for nu in range(4):
        if G[mu,nu] != 0:
            print(f"G[{coords[mu]},{coords[nu]}] = {G[mu,nu]}")

# Sample Output (Reference)
# G[t,t] = 1.0*(0.5*r**2*Derivative(Psi(r), r)**2 - 1.0*r*Psi(r)*Derivative(Psi(r), r) - 1.0*Psi(r)**2 + 1.0*Psi(r) - 0.5*Psi(r)/sin(theta)**2)/r**2
# G[r,r] = 0.5*Derivative(Psi(r), r)**2/Psi(r)**2 + 1.0*Derivative(Psi(r), r)/(r*Psi(r)) + 1.0/r**2 - 1.0/(r**2*Psi(r)) + 0.5/(r**2*Psi(r)*sin(theta)**2)
# G[theta,theta] = 0.5*r**2*Derivative(Psi(r), (r, 2)) - 0.5 + 0.5/tan(theta)**2
# G[phi,phi] = 0.5*r**2*sin(theta)**2*Derivative(Psi(r), (r, 2)) - 0.5*cos(2*theta)

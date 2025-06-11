import sympy as sp

# Define coordinates and symbols
t, r, theta, phi = sp.symbols('t r theta phi')
Psi = sp.Function('Psi')(r)

# Define the metric tensor in spherical coordinates
g = sp.Matrix([
    [-Psi,           0,              0,                    0],
    [0,     1/Psi,              0,                    0],
    [0,            0,         r**2,                    0],
    [0,            0,              0,      r**2 * sp.sin(theta)**2]
])

coords = [t, r, theta, phi]
dim = 4

# Inverse metric
g_inv = g.inv()

# Christoffel symbols
def christoffel(g, g_inv, coords):
    Gamma = {}
    for l in range(dim):
        for m in range(dim):
            for n in range(dim):
                term = 0
                for k in range(dim):
                    term += g_inv[l, k] * (
                        sp.diff(g[k, m], coords[n]) +
                        sp.diff(g[k, n], coords[m]) -
                        sp.diff(g[m, n], coords[k])
                    )
                Gamma[(l, m, n)] = sp.simplify(0.5 * term)
    return Gamma

# Compute Ricci tensor
def ricci_tensor(Gamma, coords):
    R = {}
    for m in range(dim):
        for n in range(dim):
            term = 0
            for l in range(dim):
                term += (
                    sp.diff(Gamma[(l, m, n)], coords[l])
                    - sp.diff(Gamma[(l, m, l)], coords[n])
                    + sum(Gamma[(l, m, k)] * Gamma[(k, l, n)] - Gamma[(l, n, k)] * Gamma[(k, l, m)]
                          for k in range(dim))
                )
            R[(m, n)] = sp.simplify(term)
    return R

# Compute Ricci scalar
def ricci_scalar(R, g_inv):
    R_scalar = 0
    for m in range(dim):
        for n in range(dim):
            R_scalar += g_inv[m, n] * R[(m, n)]
    return sp.simplify(R_scalar)

# Compute Einstein tensor
def einstein_tensor(R, R_scalar, g):
    G = {}
    for m in range(dim):
        for n in range(dim):
            G[(m, n)] = sp.simplify(R[(m, n)] - 0.5 * g[m, n] * R_scalar)
    return G

# Run calculations
Gamma = christoffel(g, g_inv, coords)
R = ricci_tensor(Gamma, coords)
R_scalar = ricci_scalar(R, g_inv)
G = einstein_tensor(R, R_scalar, g)

# Example output
print("Einstein Tensor Components:")
for (m, n), val in G.items():
    print(f"G[{coords[m]},{coords[n]}] =", val)


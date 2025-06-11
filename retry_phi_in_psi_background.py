
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import os

# --- GRID SETTINGS ---
r_max = 10.0
N = 600
r = np.linspace(0.5, r_max, N)
dr = r[1] - r[0]
t_max = 6.0
dt = 0.02

# --- BACKGROUND METRIC FIELD Ψ(r) ---
def Psi(r):
    return 1 - 1.9 / (r + 0.9)  # Gentle enough to curve, but allow instability

# --- INITIAL WAVE PACKET φ(r) ---
def gaussian(r, r0=3.0, width=0.5):
    return np.exp(-((r - r0)**2) / (2 * width**2))

phi0 = gaussian(r)
dphi0 = np.zeros_like(r)

# --- EVOLUTION EQUATIONS FOR φ AND ∂φ/∂t ---
def rhs(t, y):
    phi, dphi = y[:N], y[N:]
    d2phi = np.zeros_like(phi)

    for i in range(1, N - 1):
        psi = Psi(r[i])
        dpsi = (Psi(r[i + 1]) - Psi(r[i - 1])) / (2 * dr)
        d2phi[i] = (
            psi * (phi[i + 1] - 2 * phi[i] + phi[i - 1]) / dr**2 +
            dpsi * (phi[i + 1] - phi[i - 1]) / (2 * dr) +
            (2 / r[i]) * psi * (phi[i + 1] - phi[i - 1]) / (2 * dr)
        )

    return np.concatenate((dphi, d2phi))

# --- SOLVE OVER TIME ---
y0 = np.concatenate((phi0, dphi0))
sol = solve_ivp(rhs, (0, t_max), y0, t_eval=np.arange(0, t_max, dt), method='RK45', rtol=1e-6, atol=1e-9)

# --- PLOT + ANIMATION ---
fig, ax = plt.subplots()
line, = ax.plot(r, sol.y[:N, 0])
ax.set_ylim(-2, 2)
ax.set_xlim(r[0], r[-1])
ax.set_xlabel("r")
ax.set_ylabel("φ(t, r)")
title = ax.set_title("")

def update(frame):
    line.set_ydata(sol.y[:N, frame])
    title.set_text(f"Time: t = {sol.t[frame]:.2f}")
    return line, title

ani = animation.FuncAnimation(fig, update, frames=len(sol.t), blit=True)

# --- EXPORT VIDEO ---
output_path = "phi_propagation.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=2000)
ani.save(output_path, writer=writer)
print(f"Saved to: {os.path.abspath(output_path)}")

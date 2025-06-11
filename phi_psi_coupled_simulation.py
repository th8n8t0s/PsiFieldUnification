import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Simulation parameters
nx = 400
nt = 600
dx = 0.05
dt = 0.01
x = np.linspace(0, (nx - 1) * dx, nx)

# Physical constants
m = 1.0
gamma = 0.5
lambda_ = 0.2

# Initial fields
phi = np.exp(-100 * (x - 10)**2)
phi_new = np.copy(phi)
phi_old = np.copy(phi)
phi_t = np.zeros_like(x)

Psi = 1.0 - 0.8 * np.exp(-0.2 * (x - 10)**2)
Psi_new = np.copy(Psi)

# V'(phi)
def V_prime(phi):
    return m**2 * phi

# Store frames
frames = []

# Time evolution loop
for t_step in range(nt):
    phi_xx = (np.roll(phi, -1) - 2 * phi + np.roll(phi, 1)) / dx**2
    phi_tt = (Psi * phi_xx - V_prime(phi) + gamma * Psi * phi * 2 +
              np.gradient(Psi, dx) * np.gradient(phi, dx)) / Psi

    phi_new = 2 * phi - phi_old + dt**2 * phi_tt

    phi_r = np.gradient(phi, dx)
    phi_t = (phi - phi_old) / dt
    Psi_rhs = -gamma * phi**2 + 2 * lambda_ * Psi + 0.5 * (phi_r**2 - phi_t**2)
    Psi_new = Psi + 0.01 * Psi_rhs * dt

    phi_old = np.copy(phi)
    phi = np.copy(phi_new)
    Psi = np.copy(Psi_new)

    if t_step % 4 == 0:
        frames.append((phi.copy(), Psi.copy()))

# Animation
fig, ax = plt.subplots()
line1, = ax.plot(x, frames[0][0], label='φ')
line2, = ax.plot(x, frames[0][1], label='Ψ')
ax.set_ylim(-2, 2)
ax.legend()

def update(frame):
    phi_vals, Psi_vals = frame
    line1.set_ydata(phi_vals)
    line2.set_ydata(Psi_vals)
    return line1, line2

ani = FuncAnimation(fig, update, frames=frames, interval=50)

# Save output
writer = FFMpegWriter(fps=20)
ani.save("phi_psi_coupled_simulation.mp4", writer=writer)


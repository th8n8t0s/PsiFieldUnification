import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Parameters
L = 100  # spatial domain size
T = 200  # time steps
dx = 0.5
dt = 0.1
x = np.arange(-L, L, dx)
nx = len(x)

# Fields
phi = np.zeros((T, nx))
A = np.zeros((T, nx))
psi1 = np.zeros((T, nx))
psi2 = np.zeros((T, nx))

# Initial conditions
phi[0, nx//2] = 1
A[0, nx//2] = 0.5
psi1[0, nx//2] = 0.1
psi2[0, nx//2] = -0.1

# Psi background field
Psi = 1.0 / (1.0 + (x / 10)**2)

# Time evolution
for t in range(1, T-1):
    for i in range(1, nx-1):
        phi[t+1, i] = (2*phi[t, i] - phi[t-1, i] +
                       dt**2 / dx**2 * Psi[i] * (phi[t, i+1] - 2*phi[t, i] + phi[t, i-1]) -
                       dt**2 * Psi[i] * phi[t, i])
        A[t+1, i] = (2*A[t, i] - A[t-1, i] +
                     dt**2 / dx**2 * Psi[i]**2 * (A[t, i+1] - 2*A[t, i] + A[t, i-1]))
        psi1[t+1, i] = psi1[t, i] + dt * (-(psi2[t, i+1] - psi2[t, i-1]) / (2*dx)) * Psi[i]
        psi2[t+1, i] = psi2[t, i] + dt * (-(psi1[t, i+1] - psi1[t, i-1]) / (2*dx)) * Psi[i]

# Create animation
fig, ax = plt.subplots()
line1, = ax.plot(x, phi[0], label='ϕ')
line2, = ax.plot(x, A[0], label='A')
line3, = ax.plot(x, psi1[0], label='ψ₁')
line4, = ax.plot(x, psi2[0], label='ψ₂')
ax.set_ylim(-2, 2)
ax.legend()

def animate(t):
    line1.set_ydata(phi[t])
    line2.set_ydata(A[t])
    line3.set_ydata(psi1[t])
    line4.set_ydata(psi2[t])
    return line1, line2, line3, line4

ani = FuncAnimation(fig, animate, frames=T, interval=50)
output_path = "unified_field_dynamics.mp4"
ani.save(output_path, writer=animation.FFMpegWriter(fps=20))


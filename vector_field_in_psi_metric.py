# Vector Field Coupling in Psi-Defined Curved Spacetime
# Simulates a simplified Maxwell-like field in a static Psi(r)-based spherical metric background

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Spatial grid
dr = 0.01
r = np.arange(0.01, 10, dr)

# Psi background field (static Gaussian potential well)
Psi = np.exp(-((r - 5)**2))
Psi_prime = np.gradient(Psi, dr)
Psi_double_prime = np.gradient(Psi_prime, dr)

# Define initial vector potential A_t(r), A_r(r) as 1D fields
A_t = np.exp(-((r - 4.5)**2) * 5)  # Localized pulse
A_r = np.zeros_like(r)

# Field derivatives
A_t_prime = np.gradient(A_t, dr)
A_t_double_prime = np.gradient(A_t_prime, dr)

# Time evolution setup
dt = 0.005
time_steps = 500
c = 1.0  # natural units

# Prepare animation
fig, ax = plt.subplots()
line, = ax.plot(r, A_t, label='A_t(r, t)')
ax.set_ylim(-1, 1)
ax.set_title("Vector Field in Ψ-Metric Background")
ax.set_xlabel("r")
ax.set_ylabel("A_t")

# Time stepping using wave equation modified by Psi curvature
A_t_old = A_t.copy()
A_t_new = A_t.copy()

field_history = []

for _ in range(time_steps):
    curvature_term = (Psi_double_prime / Psi) - (Psi_prime**2 / Psi**2)
    wave_term = np.gradient(np.gradient(A_t, dr), dr)
    A_t_next = 2*A_t - A_t_old + dt**2 * (wave_term + curvature_term * A_t)

    A_t_old = A_t.copy()
    A_t = A_t_next.copy()
    field_history.append(A_t.copy())

# Animate the result
field_history = np.array(field_history)

def update(frame):
    line.set_ydata(field_history[frame])
    ax.set_title(f"A_t(r, t), t = {frame*dt:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=len(field_history), interval=30)

# Save the animation
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=30)
ani.save("vector_field_in_psi_background.mp4", writer=writer)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 100
T = 200
dx = 1.0
dt = 0.1
x = np.linspace(0, L, int(L/dx))
t = np.linspace(0, T, int(T/dt))
Nx = len(x)
Nt = len(t)

# Ψ curvature profile (Gaussian well)
def Psi(x):
    return 1.0 - 0.8 * np.exp(-0.01 * (x - L/2)**2)

# Initialize entangled fields φ1 and φ2
phi1 = np.zeros((Nt, Nx))
phi2 = np.zeros((Nt, Nx))

# Initial condition: symmetric Gaussian pulses
phi1[0] = np.exp(-0.1 * (x - L/3)**2)
phi2[0] = np.exp(-0.1 * (x - 2*L/3)**2)

# First time step (assume initial velocity = 0)
phi1[1] = phi1[0]
phi2[1] = phi2[0]

# Time evolution (finite difference with Ψ(x) modulation)
for n in range(1, Nt-1):
    for i in range(1, Nx-1):
        psi_val = Psi(x[i])
        phi1[n+1, i] = (2 * phi1[n, i] - phi1[n-1, i] +
                        psi_val * dt**2 / dx**2 * (phi1[n, i+1] - 2*phi1[n, i] + phi1[n, i-1]))
        phi2[n+1, i] = (2 * phi2[n, i] - phi2[n-1, i] +
                        psi_val * dt**2 / dx**2 * (phi2[n, i+1] - 2*phi2[n, i] + phi2[n, i-1]))

# Compute coherence metric: cross-correlation
coherence = np.array([np.correlate(phi1[n], phi2[n], mode='valid')[0] for n in range(Nt)])

# Plot coherence over time
plt.plot(t, coherence / np.max(coherence))  # Normalize for clarity
plt.xlabel("Time")
plt.ylabel("Normalized Cross-Correlation")
plt.title("Entanglement Coherence in Ψ Field Background")
plt.grid()
plt.tight_layout()
plt.savefig("entanglement_coherence_plot.png")

# Create animation
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='ϕ₁')
line2, = ax.plot([], [], label='ϕ₂')
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.legend()

def animate(n):
    line1.set_data(x, phi1[n])
    line2.set_data(x, phi2[n])
    return line1, line2

ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=30)
ani.save("entangled_field_evolution.mp4", writer='ffmpeg')


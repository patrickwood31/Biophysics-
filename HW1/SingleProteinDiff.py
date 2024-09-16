import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 1000  # DNA length in nm
a = 1     # step size in nm
N = L // a  # number of discrete positions
D_slide = 1e4  # sliding diffusion constant in nm^2/s
w = D_slide / a**2  # transition rate
dt = 1e-5  # time step in seconds
tau_3D = 8  # 3D search time in seconds
tau_3D_jump = tau_3D / N  # time for a single 3D jump

# Part a: Simulate 1D diffusion and calculate MSD
def simulate_diffusion(t_end, i0=N//2):
    """Simulate 1D diffusion and return the final position."""
    i = i0
    t = 0
    while t < t_end:
        r = np.random.random()
        # System Dynamics 
        if r < w * dt:
            i += 1
        elif r < 2 * w * dt:
            i -= 1
        
        # Implement periodic boundary conditions (DNA loop)
        if i > N:
            i = 1
        elif i < 1:
            i = N
        
        t += dt
    return i

def calculate_msd(t_end, num_trajectories=1000):
    """Calculate mean squared displacement for a given end time."""
    displacements = []
    for _ in range(num_trajectories):
        final_i = simulate_diffusion(t_end)
        # Calculate displacement considering periodic boundary conditions
        displacement = min((final_i - N//2) % N, (N//2 - final_i) % N)
        displacements.append(displacement**2)
    return a**2 * np.mean(displacements)

# Calculate MSD for different end times
t_values = np.logspace(-3, 0, 20)
msd_values = [calculate_msd(t) for t in t_values]

# Plot MSD vs time as requested in part a
plt.figure(figsize=(10, 6))
plt.loglog(t_values, msd_values, 'bo-', label='Simulation')
plt.loglog(t_values, 2 * D_slide * t_values, 'r--', label='Theory: 2Dt')
plt.xlabel('Time (s)')
plt.ylabel('Mean Squared Displacement (nm^2)')
plt.legend()
plt.title('1D Diffusion: Mean Squared Displacement vs Time')
plt.grid(True)
plt.show()

print(f"MSD at t=1s: {msd_values[-1]:.2f} nm^2")
print(f"Theoretical MSD at t=1s: {2 * D_slide:.2f} nm^2")

# Part b: Simulate 1D search for a target
def simulate_search():
    """Simulate 1D search for a target and return the search time."""
    # Choose random starting position as specified in part b
    i = np.random.randint(1, N+1)
    t = 0
    while True:
        # Check if target is reached (i=1 or i=N+1 due to periodic boundary)
        if i == 1 or i == N+1:
            return t
        
        r = np.random.random()
        if r < w * dt:
            i += 1
        elif r < 2 * w * dt:
            i -= 1
        
        # Implement periodic boundary conditions
        if i > N:
            i = 1
        elif i < 1:
            i = N
        
        t += dt

# Perform multiple searches and calculate average search time (τ_1D)
num_searches = 1000
search_times = [simulate_search() for _ in range(num_searches)]
tau_1D = np.mean(search_times)

print(f"Average 1D search time (τ_1D): {tau_1D:.6f} s")

# Part c: Analytical solution for τ_1D
def analytical_tau_1D():
    """Calculate analytical τ_1D as derived in part c."""
    x = np.linspace(0, L, 1000)
    tau_x = (x * (L - x)) / (2 * D_slide)
    return np.mean(tau_x)

tau_1D_analytical = analytical_tau_1D()
print(f"Analytical 1D search time: {tau_1D_analytical:.6f} s")

# Parts d-h: Implement 1D+3D search strategy
def simulate_search_1D_3D(gamma):
    """Simulate 1D+3D search for a target and return the search time."""
    i = np.random.randint(1, N+1)  # random starting position
    t = 0
    while True:
        if i == 1 or i == N+1:  # target found
            return t
        
        r = np.random.random()
        # Implement the dynamics described in part d
        if r < w * dt:
            i += 1
        elif r < 2 * w * dt:
            i -= 1
        elif r < (2 * w + gamma) * dt:
            # 3D jump: reset position randomly and add τ_3D_jump to time
            i = np.random.randint(1, N+1)
            t += tau_3D_jump
        
        # Implement periodic boundary conditions
        if i > N:
            i = 1
        elif i < 1:
            i = N
        
        t += dt

def calculate_tau_1D_3D(gamma, num_searches=1000):
    """Calculate average search time for a given gamma."""
    search_times = [simulate_search_1D_3D(gamma) for _ in range(num_searches)]
    return np.mean(search_times)

# Calculate τ_1D+3D for different γ values as requested in part d
gamma_values = [0, 1, 10, 100, 1000]
tau_1D_3D_values = [calculate_tau_1D_3D(gamma) for gamma in gamma_values]

# Print results
for gamma, tau in zip(gamma_values, tau_1D_3D_values):
    print(f"γ = {gamma} s^-1: τ_1D+3D = {tau:.6f} s")

# Plot τ_1D+3D vs γ
plt.figure(figsize=(10, 6))
plt.semilogx(gamma_values, tau_1D_3D_values, 'bo-')
plt.xlabel('γ (s^-1)')
plt.ylabel('τ_1D+3D (s)')
plt.title('Search Time vs Dissociation Rate')
plt.grid(True)
plt.show()

# Parts e-h: Analytical model for 1D+3D search
def analytical_tau_1D_3D(gamma):
    """
    Calculate analytical τ_1D+3D based on the derivation in parts e-h.
    
    e) t_slide = 1 / gamma (average time spent sliding)
    f) ρ ≈ sqrt(2D_slide * t_slide) / L (probability of finding target during one slide)
    g) M = 1 / ρ (average number of slide+jump cycles)
    h) τ_1D+3D = M * (t_slide + τ_3D_jump)
    """
    t_slide = 1 / gamma
    rho = np.sqrt(2 * D_slide * t_slide) / L
    M = 1 / rho
    return M * (t_slide + tau_3D_jump)

# Calculate analytical τ_1D+3D for a range of γ values
gamma_range = np.logspace(-1, 3, 100)
tau_1D_3D_analytical = [analytical_tau_1D_3D(gamma) for gamma in gamma_range]

# Find the optimal γ that minimizes τ_1D+3D
optimal_gamma = gamma_range[np.argmin(tau_1D_3D_analytical)]
min_tau = np.min(tau_1D_3D_analytical)

print(f"Optimal γ: {optimal_gamma:.2f} s^-1")
print(f"Minimum τ_1D+3D: {min_tau:.6f} s")

# Plot analytical τ_1D+3D vs γ and compare with simulation results
plt.figure(figsize=(10, 6))
plt.loglog(gamma_range, tau_1D_3D_analytical, 'r-', label='Analytical')
plt.loglog(gamma_values, tau_1D_3D_values, 'bo', label='Simulation')
plt.axvline(optimal_gamma, color='g', linestyle='--', label='Optimal γ')
plt.xlabel('γ (s^-1)')
plt.ylabel('τ_1D+3D (s)')
plt.title('Search Time vs Dissociation Rate: Analytical vs Simulation')
plt.legend()
plt.grid(True)
plt.show()
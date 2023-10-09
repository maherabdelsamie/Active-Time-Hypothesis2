import numpy as np
import matplotlib.pyplot as plt
import csv


# Define a custom normal distribution generator using Box-Muller transform
def random_normal(mean=0.0, std_dev=1.0):
    u1 = np.random.rand()
    u2 = np.random.rand()
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return mean + std_dev * z0

def create_complex_Hamiltonian(size):
    real_part = np.random.rand(size, size)
    imag_part = np.random.rand(size, size)
    return real_part + 1j * imag_part

def U(dt, H):
    return np.exp(-1j * H * dt)

def non_adjacent_state_entropy(psi):
    rho = np.outer(psi, psi.conj())
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return entropy

def evolve_with_time(psi, time_state):
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("State collapsed to zero!")
    psi /= norm
    return psi

def simulate_time_operator(psi):
    return np.std(np.abs(psi)**2)

def measure_state_prediction_dev(psi, psi_benchmark):
    overlap = np.abs(np.dot(psi.conj(), psi_benchmark))
    overlap = max(0, min(1, overlap))
    return 1 - overlap**2


# Generative Function S(t)
def S(t):
    # Introduce unpredictability in time using random noise
    return random_normal(mean=0, std_dev=0.01)  # Decreased std_dev to 0.01

# Directive Function G(Phi, t)
def G(Phi, t):
    k = 0.01  # Constant to control the feedback strength
    return -k * np.abs(Phi)

# Adaptive Function A(Phi)
def A(Phi):
    m = 0.01  # Constant to control the adaptive rate
    return m * np.abs(Phi)

def run_simulation(use_active_time=True):
    results = []

    N_particles = 2
    dim = 3
    psi_system = np.random.rand(dim**N_particles) + 1j * np.random.rand(dim**N_particles)
    psi_system /= np.linalg.norm(psi_system)
    H_system = create_complex_Hamiltonian(dim**N_particles)
    
    T = 10
    dt = 0.01
    time_0_state = 0.9
    time_1_state = 0.1
    time_state = (time_0_state + time_1_state) / np.sqrt(2)
    psi_benchmark = psi_system.copy()

    def stochastic_dt(dt, psi):
        if use_active_time:
            adaptive_term = A(psi)
            dt += random_normal(mean=0, std_dev=0.1*dt)
            dt += adaptive_term  # Adding the adaptive term
            return dt
        else:
            return dt

    for t in np.arange(0, T, dt):
        dt = stochastic_dt(dt, psi_system)
        dPhi_dtau = (1 + A(psi_system)) * (S(t) + G(psi_system, t))
        psi_system += dPhi_dtau * dt
        psi_system = evolve_with_time(psi_system, time_state)
        psi_system = U(dt, H_system) @ psi_system
        entropy = non_adjacent_state_entropy(psi_system)
        time_uncertainty = simulate_time_operator(psi_system)
        prediction_error = measure_state_prediction_dev(psi_system, psi_benchmark)
        
        # Ensure normalization
        psi_system /= np.linalg.norm(psi_system)
        
        results.append((entropy, time_uncertainty, prediction_error))

    return results


# Function to save results to a CSV file
def save_to_csv(data, filename="simulation_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers to CSV file
        writer.writerow(["Entropy", "Time Uncertainty", "Prediction Error"])
        # Write data to CSV file
        for row in data:
            writer.writerow(row)

# Save the results
save_to_csv(results_with_active_time, "results_with_active_time.csv")
save_to_csv(results_without_active_time, "results_without_active_time.csv")

# Running the simulations
results_with_active_time = run_simulation(True)
results_without_active_time = run_simulation(False)

# Extract data for plotting
entropies_with = [item[0] for item in results_with_active_time]
time_uncertainties_with = [item[1] for item in results_with_active_time]
prediction_errors_with = [item[2] for item in results_with_active_time]

entropies_without = [item[0] for item in results_without_active_time]
time_uncertainties_without = [item[1] for item in results_without_active_time]
prediction_errors_without = [item[2] for item in results_without_active_time]

# Timepoints for plotting
timepoints = np.arange(0, 10, 0.01)

# Plot entropies
plt.figure(figsize=(12, 8))
plt.plot(timepoints, entropies_with, label="With Active Time", color="blue")
plt.plot(timepoints, entropies_without, label="Without Active Time", color="red", linestyle='--')
plt.xlabel('Time')
plt.ylabel('Entropy')
plt.title('Entropy Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot time uncertainties
plt.figure(figsize=(12, 8))
plt.plot(timepoints, time_uncertainties_with, label="With Active Time", color="blue")
plt.plot(timepoints, time_uncertainties_without, label="Without Active Time", color="red", linestyle='--')
plt.xlabel('Time')
plt.ylabel('Time Uncertainty')
plt.title('Time Uncertainty Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot prediction errors
plt.figure(figsize=(12, 8))
plt.plot(timepoints, prediction_errors_with, label="With Active Time", color="blue")
plt.plot(timepoints, prediction_errors_without, label="Without Active Time", color="red", linestyle='--')
plt.xlabel('Time')
plt.ylabel('Prediction Error')
plt.title('Prediction Error Over Time')
plt.legend()
plt.grid(True)
plt.show()


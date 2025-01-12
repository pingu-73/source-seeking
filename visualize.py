from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

from main import num_particles, dim, apso, spso, arpso

def visualize_simulation(algorithm, src_position=np.array([50, 50]), num_frames=100):
    global positions, velocities, accelerations

    positions = np.random.uniform(0, 100, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    accelerations = np.zeros((num_particles, dim))
    
    positions_history = [positions.copy()]
    for _ in range(num_frames):
        if algorithm == "APSO":
            apso(src_position)
        elif algorithm == "SPSO":
            spso(src_position)
        elif algorithm == "ARPSO":
            arpso(src_position)
        positions_history.append(positions.copy())
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"{algorithm} Simulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    particles, = ax.plot([], [], 'bo', label="UAVs")
    source, = ax.plot([], [], 'ro', label="Source")
    ax.legend()

    def update(frame):
        current_positions = positions_history[frame]
        particles.set_data(current_positions[:, 0], current_positions[:, 1])
        source.set_data(src_position[0], src_position[1])
        return particles, source
    
    anim = FuncAnimation(fig, update, frames=len(positions_history), interval=200, blit=True)
    plt.show()


def performance_vs_uavs(algorithm, uav_counts, src_position=np.array([50, 50])):
    avg_iterations = []
    for count in uav_counts:
        global num_particles, positions, velocities, accelerations
        num_particles = count
        positions = np.random.uniform(0, 100, (num_particles, dim))
        velocities = np.zeros((num_particles, dim))
        accelerations = np.zeros((num_particles, dim))
        
        if algorithm == "APSO":
            results = [apso(src_position) for _ in range(100)]
        elif algorithm == "SPSO":
            results = [spso(src_position) for _ in range(100)]
        elif algorithm == "ARPSO":
            results = [arpso(src_position) for _ in range(100)]
        
        iterations, _ = zip(*results)
        avg_iterations.append(np.mean(iterations))
    return avg_iterations


def swarm_distance_vs_uavs(algorithm, uav_counts, src_position=np.array([50, 50])):
    avg_distances = []
    for count in uav_counts:
        global num_particles, positions, velocities, accelerations
        num_particles = count
        positions = np.random.uniform(0, 100, (num_particles, dim))
        velocities = np.zeros((num_particles, dim))
        accelerations = np.zeros((num_particles, dim))

        if algorithm == "APSO":
            results = [apso(src_position) for _ in range(100)]
        elif algorithm == "SPSO":
            results = [spso(src_position) for _ in range(100)]
        elif algorithm == "ARPSO":
            results = [arpso(src_position) for _ in range(100)]

        _, distances = zip(*results)
        avg_distances.append(np.mean(distances))

    return avg_distances


# visualize_simulation("APSO")
# visualize_simulation("SPSO")
# visualize_simulation("ARPSO")



uav_counts = [5, 10, 15, 20, 25, 30]
apso_iterations = performance_vs_uavs("APSO", uav_counts)
spso_iterations = performance_vs_uavs("SPSO", uav_counts)
arpso_iterations = performance_vs_uavs("ARPSO", uav_counts)
plt.figure(figsize=(10, 6))
plt.plot(uav_counts, apso_iterations, label="APSO", marker='o')
plt.plot(uav_counts, spso_iterations, label="SPSO", marker='s')
plt.plot(uav_counts, arpso_iterations, label="ARPSO", marker='^')
plt.xlabel("Number of UAVs")
plt.ylabel("Average Number of Iterations")
plt.title("Number of UAVs vs. Average Number of Iterations")
plt.legend()
plt.grid()
plt.show()


apso_distances = swarm_distance_vs_uavs("APSO", uav_counts)
spso_distances = swarm_distance_vs_uavs("SPSO", uav_counts)
arpso_distances = swarm_distance_vs_uavs("ARPSO", uav_counts)
plt.figure(figsize=(10, 6))
plt.plot(uav_counts, apso_distances, label="APSO", marker='o')
plt.plot(uav_counts, spso_distances, label="SPSO", marker='s')
plt.plot(uav_counts, arpso_distances, label="ARPSO", marker='^')
plt.xlabel("Number of UAVs")
plt.ylabel("Average Swarm Distance")
plt.title("Effect of Number of UAVs on Average Swarm Distance")
plt.legend()
plt.grid()
plt.show()
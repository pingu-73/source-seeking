import matplotlib.pyplot as plt
import numpy as np

def fitness_signal_strength(pos, src_pos, src_pow = 100, a = 0.1, noise = 0.0):
    distance = np.linalg.norm(pos - src_pos)
    signal_strength = src_pow * np.exp(-a*distance**2)
    rnd_noise = noise * np.random.uniform(-1, 1)
    return signal_strength + rnd_noise



class APSO:
    def __init__(self, c1=1.193, c2=1.193, w1=0.675, w2=-0.285, num_particles=5, dim=2, max_iter=1000, T=1, threshold=0.1):
        self.c1 = c1
        self.c2 = c2
        self.w1 = w1
        self.w2 = w2
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.T = T
        self.threshold_dis = threshold

        self.positions = np.random.uniform(0, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.accelerations = np.zeros((num_particles, dim))


    def update_acceleration(self, accelerations, positions, p_best, g_best):
        r1 = np.random.uniform(0, self.c1, (self.num_particles, self.dim))
        r2 = np.random.uniform(0, self.c2, (self.num_particles, self.dim))
        accelerations = self.w1 * accelerations + r1 * (p_best - positions) + r2 * (g_best - positions)
        return accelerations


    def update_velocity(self, velocities, accelerations):
        velocities = self.w2 * velocities + accelerations * self.T
        return velocities


    def update_position(self, positions, velocities):
        positions = positions + velocities * self.T
        return positions


    def apso(self, src_position=np.array([80, 34])):
        b_positions = self.positions.copy()
        g_position = self.positions[np.argmax([fitness_signal_strength(pos, src_position) for pos in self.positions])]
        total_distance = 0

        for i in range(self.max_iter):
            self.accelerations = self.update_acceleration(self.accelerations, self.positions, b_positions, g_position)
            self.velocities = self.update_velocity(self.velocities, self.accelerations)
            self.positions = self.update_position(self.positions, self.velocities)

            total_distance += np.sum(np.linalg.norm(self.velocities, axis=1))

            fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in self.positions])

            for j in range(self.num_particles):
                if fitness_values[j] > fitness_signal_strength(b_positions[j], src_position):
                    b_positions[j] = self.positions[j]
            g_position = self.positions[np.argmax(fitness_values)]

            if np.min([np.linalg.norm(pos - src_position) for pos in self.positions]) < self.threshold_dis:
                # print(f"src found in {i + 1} iterations")
                return i + 1, total_distance

        # print(f"src not found within {self.max_iter} iterations.")
        return self.max_iter, total_distance
    





class SPSO:
    def __init__(self, c1 = 1.193, c2 = 1.193, w = 0.721, num_particles=5, dim=2, max_iter=1000, threshold = 0.1):
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.num_particles = num_particles
        self.dim = dim
        self.dim = dim
        self.max_iter = max_iter
        self.threshold_dis = threshold

        self.positions = np.random.uniform(0, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.full(num_particles, -np.inf)


    def update_velocity(self, velocities, positions, p_best, g_best):
        r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        velocities = self.w * velocities +  r1*self.c1*(p_best - positions) + r2*self.c2*(g_best - positions)
        return velocities


    def update_positions(self, velocities, positions):
        positions = positions + velocities
        return positions
    

    def spso(self, src_position=np.array([80, 34])):
        g_position = None
        g_score = -np.inf
        total_distance = 0

        for i in range(self.max_iter):
            fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in self.positions])

            for j in range(self.num_particles):
                if fitness_values[j] > self.b_scores[j]:
                    self.b_scores[j] = fitness_values[j]
                    self.b_positions[j] = self.positions[j]
            
            best_particle = np.argmax(fitness_values)
            
            if fitness_values[best_particle] > g_score:
                g_score = fitness_values[best_particle]
                g_position = self.positions[best_particle]
            
            self.velocities = self.update_velocity(self.velocities, self.positions, self.b_positions, g_position)
            self.positions = self.update_positions(self.positions, self.velocities)

            total_distance += np.sum(np.linalg.norm(self.velocities, axis=1))

            if np.min([np.linalg.norm(pos - src_position) for pos in self.positions]) < self.threshold_dis:
                    # print(f"src found in {i + 1} iterations")
                    return i + 1, total_distance 

        # print(f"src not found within {self.max_iter} iterations.")
        return self.max_iter, total_distance
    

class ARPSO:
    def __init__(self, c1=1.193, c2=1.193, c3=0, num_particles=5, dim=2, max_iter=1000, threshold=0.1, sensing_radius=10):
        self.c1 = c1
        self.c2 = c2
        self.c3 = 0
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.threshold_dis = threshold
        self.sensing_radius = sensing_radius

        self.positions = np.random.uniform(0, 100, (num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.full(num_particles, -np.inf)

    def calculate_inertia_weight(self, fitness_values, iteration, max_iter):
        evolutionary_speed = 1 - (min(fitness_values) / max(fitness_values))
        aggregation_degree = min(fitness_values) / max(fitness_values)
        return 1 * (1 - 0.5 * evolutionary_speed + 0.5 * aggregation_degree)

    def calculate_c3(self, positions, obstacles):
        c3 = np.zeros(self.num_particles)
        for i, pos in enumerate(positions):
            distance_to_obstacles = np.linalg.norm(obstacles - pos, axis=1)
            if np.any(distance_to_obstacles < self.sensing_radius):
                c3[i] = 2 * self.c1 + self.c2
        return c3

    def update_velocity(self, velocities, positions, p_best, g_best, attractive_pos, w, c3_values):
        r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        r3 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        velocities = (
            w[:, np.newaxis] * velocities +
            self.c1 * r1 * (p_best - positions) +
            self.c2 * r2 * (g_best - positions) +
            c3_values[:, np.newaxis] * r3 * (attractive_pos - positions)
        )
        return velocities

    def update_position(self, positions, velocities):
        positions += velocities
        return positions

    def arpso(self, src_position=np.array([80, 34]), obstacles=np.array([[50, 50]])):
        g_position = None
        g_score = -np.inf
        total_distance = 0

        attractive_position = np.random.uniform(0, 100, (self.num_particles, self.dim))

        for i in range(self.max_iter):
            fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in self.positions])
            
            w = np.array([self.calculate_inertia_weight(fitness_values, i, self.max_iter) for _ in range(self.num_particles)])
            
            for j in range(self.num_particles):
                if fitness_values[j] > self.b_scores[j]:
                    self.b_scores[j] = fitness_values[j]
                    self.b_positions[j] = self.positions[j]

            best_particle = np.argmax(fitness_values)
            if fitness_values[best_particle] > g_score:
                g_score = fitness_values[best_particle]
                g_position = self.positions[best_particle]

            c3_values = self.calculate_c3(self.positions, obstacles)

            self.velocities = self.update_velocity(self.velocities, self.positions, self.b_positions, g_position, attractive_position, w, c3_values)
            self.positions = self.update_position(self.positions, self.velocities)

            total_distance += np.sum(np.linalg.norm(self.velocities, axis=1))

            if np.min([np.linalg.norm(pos - src_position) for pos in self.positions]) < self.threshold_dis:
                # print(f"Source found in {i + 1} iterations!")
                return i + 1, total_distance

        # print("Source not found within the maximum iterations.")
        return self.max_iter, total_distance
    



def stability_analysis(apso_instance):
    w1, w2, c1, c2, T = apso_instance.w1, apso_instance.w2, apso_instance.c1, apso_instance.c2, apso_instance.T
    stability_1 = (2 / T) * (1 + w1 + w2 + w1 * w2) > c1 + c2
    stability_2 = abs(w1 * w2) < 1
    stability_3 = abs((1 - w1 * w2) * (w1 + w2) + (w1 * w2) * (c1 * T + c2 * T)) < abs(1 - (w1 * w2)**2)
    stability_4 = abs(w1 + w2) < abs(1 + w1 * w2)

    is_stable = stability_1 and stability_2 and stability_3 and stability_4
    print(f"Stability Analysis: {'Passed' if is_stable else 'Failed'}")


def convergence_analysis(apso_instance, src_position=np.array([80, 34])):
    global_best_fitness = []
    current_global_best = -np.inf

    for _ in range(apso_instance.max_iter):
        iterations, distance = apso_instance.apso(src_position)
        fitness_values = np.array([fitness_signal_strength(pos, src_position) for pos in apso_instance.positions])
        max_fitness = np.max(fitness_values)
        current_global_best = max(current_global_best, max_fitness)
        global_best_fitness.append(current_global_best)

    plt.figure(figsize=(10, 6))
    plt.plot(range(apso_instance.max_iter), global_best_fitness, label="Global Best Fitness")
    plt.title("Convergence Analysis of APSO", fontsize=16)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Global Best Fitness", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



class PerformanceMetrics:
    def __init__(self, algorithms, num_simulations=10, uav_counts=None, src_position=np.array([80, 34])):
        """
        Initialize the PerformanceMetrics class.

        :param algorithms: Dictionary of algorithms to evaluate (e.g., {"APSO": APSO(), "SPSO": SPSO(), "ARPSO": ARPSO()}).
        :param num_simulations: Number of Monte Carlo simulations to run.
        :param uav_counts: List of UAV counts to evaluate (e.g., [5, 10, 15, 20]).
        :param src_position: Position of the source in the search space.
        """
        self.algorithms = algorithms
        self.num_simulations = num_simulations
        self.uav_counts = uav_counts or [5, 10, 15, 20]
        self.src_position = src_position

    def evaluate(self):
        results = {alg_name: {"time": [], "iterations": [], "distance": []} for alg_name in self.algorithms.keys()}

        for num_uavs in self.uav_counts:
            for alg_name, alg in self.algorithms.items():
                alg_results = {"time": [], "iterations": [], "distance": []}

                for _ in range(self.num_simulations):
                    # Reset algorithm state for each simulation
                    alg.num_particles = num_uavs
                    alg.positions = np.random.uniform(0, 100, (num_uavs, alg.dim))
                    alg.velocities = np.zeros((num_uavs, alg.dim))
                    if hasattr(alg, "accelerations"):
                        alg.accelerations = np.zeros((num_uavs, alg.dim))
                    if hasattr(alg, "b_positions"):
                        alg.b_positions = alg.positions.copy()
                        alg.b_scores = np.full(num_uavs, -np.inf)

                    # Dynamically call the appropriate method
                    method = getattr(alg, alg_name.lower())
                    iterations, distance = method(self.src_position)

                    # Collect metrics
                    alg_results["time"].append(iterations)  # 1 iteration = 1 unit time
                    alg_results["iterations"].append(iterations)
                    alg_results["distance"].append(distance)

                # Compute averages
                results[alg_name]["time"].append(np.mean(alg_results["time"]))
                results[alg_name]["iterations"].append(np.mean(alg_results["iterations"]))
                results[alg_name]["distance"].append(np.mean(alg_results["distance"]))

        return results

    def plot_results(self, results):
        for metric in ["time", "iterations", "distance"]:
            plt.figure(figsize=(10, 6))
            for alg_name in self.algorithms.keys():
                plt.plot(self.uav_counts, results[alg_name][metric], label=alg_name)

            plt.title(f"Average {metric.capitalize()} vs Number of UAVs", fontsize=16)
            plt.xlabel("Number of UAVs", fontsize=12)
            plt.ylabel(f"Average {metric.capitalize()}", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.show()




dis_apso_list = []
dis_spso_list = []
dis_arpso_list = []
print("APSO\t\t\t SPSO\t\t\t ARPSO")
print("Iter\t Dist\t\t Iter\t Dist\t\t Iter\t Dist")
print("----------------------------------------------------------------")
for _ in range(10):
    apso_sim = APSO()
    spso_sim = SPSO()
    arpso_sim = ARPSO()
    iterations_apso, distance_apso = apso_sim.apso()
    iterations_spso, distance_spso = spso_sim.spso()
    iterations_arpso, distance_arpso = arpso_sim.arpso()

    dis_apso_list.append(distance_apso)
    dis_spso_list.append(distance_spso)
    dis_arpso_list.append(distance_arpso)

    # convergence_analysis(apso_sim)

    print(f"{iterations_apso:<8} {distance_apso:<8.3f}\t {iterations_spso:<8} {distance_spso:<8.3f}\t {iterations_arpso:<8} {distance_arpso:<8.3f}")



apso_sim = APSO()
spso_sim = SPSO()
arpso_sim = ARPSO()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), dis_apso_list, label='APSO Distance', marker='o')
plt.plot(range(1, 11), dis_spso_list, label='SPSO Distance', marker='s')
plt.plot(range(1, 11), dis_arpso_list, label='ARPSO Distance', marker='*')
plt.title("APSO vs SPSO vs ARPSO Performance", fontsize=16)
plt.xlabel("Iterations/Runs", fontsize=12)
plt.ylabel("Distances travelled to find the source", fontsize=12)
plt.xticks(range(1, 11))
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
# plt.savefig("spso_vs_apso_vs_arpso.png", dpi = 300)
plt.show()
plt.close()



apso_sim = APSO()
spso_sim = SPSO()
arpso_sim = ARPSO()

metrics = PerformanceMetrics(
    algorithms={"APSO": apso_sim, "SPSO": spso_sim, "ARPSO": arpso_sim},
    num_simulations=10,
    uav_counts=[5, 10, 15, 20]
)

# Evaluate and plot results
results = metrics.evaluate()
metrics.plot_results(results)


# stability_analysis(apso_sim)
# convergence_analysis(apso_sim)
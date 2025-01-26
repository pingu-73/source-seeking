import matplotlib.pyplot as plt
import numpy as np

from benchmarks import (
    ackley,
    composition_1,
    composition_2,
    griewank,
    hybrid_1,
    hybrid_2,
    quartic_with_noise,
    rastrigin,
    schwefel,
    schwefel_12,
    schwefel_222,
    sphere,
    high_conditioned_elliptic,
    schwefel_26,
    rosenbrock,
)


def fitness_signal_strength(pos, src_pos, src_pow = 100, a = 0.1, noise = 0.0):
    distance = np.linalg.norm(pos - src_pos)
    signal_strength = src_pow * np.exp(-a*distance**2)
    noise_std = noise * signal_strength
    rnd_noise = np.random.normal(0, noise_std)
    # rnd_noise = noise * np.random.uniform(-1, 1)
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

    def reset_algorithm_state(self):
        self.positions = np.random.uniform(0, 100, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.accelerations = np.zeros((self.num_particles, self.dim))

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


    def reset_algorithm_state(self):
        self.positions = np.random.uniform(0, 100, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.full(self.num_particles, -np.inf)

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

    def reset_algorithm_state(self):
        self.positions = np.random.uniform(0, 100, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.b_positions = self.positions.copy()
        self.b_scores = np.full(self.num_particles, -np.inf)

    def calculate_inertia_weight(self, fitness_values, iteration, max_iter):
        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)
        if max_fitness == 0:
            return 1  # Default inertia weight when max fitness is zero
        evolutionary_speed = 1 - (min_fitness / max_fitness)
        aggregation_degree = min_fitness / max_fitness if max_fitness != 0 else 0
        # normalized_fitness = (fitness_values - min_fitness) / (max_fitness - min_fitness + 1e-9)
        return 1 * (1 - 0.5 * evolutionary_speed + 0.5 * aggregation_degree)
        

    # def calculate_inertia_weight(self, fitness_values, iteration, max_iter):
    #     evolutionary_speed = 1 - (min(fitness_values) / max(fitness_values))
    #     aggregation_degree = min(fitness_values) / max(fitness_values)
    #     return 1 * (1 - 0.5 * evolutionary_speed + 0.5 * aggregation_degree)

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
    # plt.savefig("./figure/Paper1/r1_with_noise/convergence_analysis.png", dpi = 300)
    plt.show()
    plt.close()



class PerformanceMetrics:
    def __init__(self, algorithms, num_simulations=10, uav_counts=None, src_position=np.array([80, 34])):
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
                    # alg.reset_algorithm_state()
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
            # plt.savefig(f"./figure/Paper1/r1_with_noise/{metric}_VS_uav.png", dpi = 300)
            plt.show()
            plt.close()





def noise_analysis(algorithms, noise_levels=[0.0, 0.05, 0.1], num_simulations=10, uav_count=10, src_position=np.array([80, 34])):
    results = {alg_name: [] for alg_name in algorithms.keys()}
    
    for noise in noise_levels:
        for alg_name, alg in algorithms.items():
            total_iterations, total_distance = 0, 0
            
            for _ in range(num_simulations):
                # Update number of particles and reset state
                alg.num_particles = uav_count
                alg.reset_algorithm_state()
                

                method = getattr(alg, alg_name.lower())
                iterations, distance = method(src_position)
                total_iterations += iterations
                total_distance += distance
            
            results[alg_name].append({
                "noise": noise,
                "avg_iterations": total_iterations / num_simulations,
                "avg_distance": total_distance / num_simulations,
            })
    
    return results



def plot_noise_results(results):
    for metric in ["avg_iterations", "avg_distance"]:
        plt.figure(figsize=(10, 6))
        for alg_name, metrics in results.items():
            noise_levels = [m["noise"] for m in metrics]
            values = [m[metric] for m in metrics]
            plt.plot(noise_levels, values, label=alg_name, marker='o')

        plt.title(f"Effect of Noise on {metric.replace('_', ' ').capitalize()}", fontsize=16)
        plt.xlabel("Noise Level", fontsize=12)
        plt.ylabel(metric.replace("_", " ").capitalize(), fontsize=12)
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
# plt.savefig("./figure/Paper1/r1_with_noise/spso_vs_apso_vs_arpso.png", dpi = 300)
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


stability_analysis(apso_sim)
convergence_analysis(apso_sim)

noise_results = noise_analysis(
    algorithms={"APSO": apso_sim, "SPSO": spso_sim, "ARPSO": arpso_sim},
    noise_levels=[0.0, 0.05, 0.1],
    num_simulations=10,
    uav_count=10,
    src_position=np.array([80, 34])
)
plot_noise_results(noise_results)


class BenchmarkPerformance:
    def __init__(self, algorithms, benchmark_functions, num_simulations=10, dim=2, threshold=0.1):
        self.algorithms = algorithms
        self.benchmark_functions = benchmark_functions
        self.num_simulations = num_simulations
        self.dim = dim
        self.threshold = threshold

    # def evaluate(self):
    #     results = {alg_name: {func_name: {"iterations": [], "distance": [], "success_rate": 0}
    #                           for func_name in self.benchmark_functions.keys()}
    #                for alg_name in self.algorithms.keys()}

    #     for func_name, func in self.benchmark_functions.items():
    #         for alg_name, alg in self.algorithms.items():
    #             success_count = 0

    #             for _ in range(self.num_simulations):
    #                 alg.positions = np.random.uniform(-100, 100, (alg.num_particles, self.dim))
    #                 alg.velocities = np.zeros((alg.num_particles, self.dim))
    #                 if hasattr(alg, "accelerations"):
    #                     alg.accelerations = np.zeros((alg.num_particles, self.dim))
    #                 if hasattr(alg, "b_positions"):
    #                     alg.b_positions = alg.positions.copy()
    #                     alg.b_scores = np.full(alg.num_particles, -np.inf)


    #                 method = getattr(alg, alg_name.lower())
    #                 iterations, distance = method(src_position=np.zeros(self.dim))


    #                 fitness_values = np.array([func(pos) for pos in alg.positions])
    #                 # debug
    #                 # print(f"Number of particles: {alg.num_particles}, positions shape: {alg.positions.shape}")
    #                 # print(f"alg.positions shape: {alg.positions.shape}, fitness_values shape: {fitness_values.shape}")

    #                 if len(fitness_values) == 0:
    #                     print(f"No fitness values for {alg_name} on {func_name}, skipping.")
    #                     continue
    #                 if fitness_values.shape[0] != alg.positions.shape[0]:
    #                     print(f"Mismatch in positions and fitness values for {alg_name} on {func_name}")
    #                     continue
                    
    #                 # print(f"Fitness Values: {fitness_values}, Shape: {fitness_values.shape}")#debug
    #                 if fitness_values.ndim != 1:
    #                     raise ValueError(f"fitness_values must be a 1D array, but got shape {fitness_values.shape}")

    #                 best_position = alg.positions[np.argmax(fitness_values)]
    #                 # print(f"Best position: {best_position}, Best fitness: {max(fitness_values)}") #debug
                    
    #                 ### @TODO:
    #                 global_minimum = func(np.zeros(self.dim))
    #                 best_fitness = func(best_position)
    #                 if abs(best_fitness - global_minimum) < self.threshold:
    #                     success_count += 1
    #                 # if np.linalg.norm(best_position) < self.threshold:
    #                 #     success_count += 1

    #                 # Collect metrics
    #                 results[alg_name][func_name]["iterations"].append(iterations)
    #                 results[alg_name][func_name]["distance"].append(distance)

    #             # Calculate success rate
    #             results[alg_name][func_name]["success_rate"] = success_count / self.num_simulations

    #     return results
    

    def evaluate(self):
        results = {alg_name: {func_name: {"iterations": [], "distance": [], "success_rate": 0}
                            for func_name in self.benchmark_functions.keys()}
                for alg_name in self.algorithms.keys()}

        for func_name, func in self.benchmark_functions.items():
            for alg_name, alg in self.algorithms.items():
                success_count = 0

                for _ in range(self.num_simulations):
                    alg.reset_algorithm_state()
                    # alg.positions = np.random.uniform(-100, 100, (alg.num_particles, self.dim))
                    # alg.velocities = np.zeros((alg.num_particles, self.dim))
                    # if hasattr(alg, "accelerations"):
                    #     alg.accelerations = np.zeros((alg.num_particles, self.dim))
                    # if hasattr(alg, "b_positions"):
                    #     alg.b_positions = alg.positions.copy()
                    #     alg.b_scores = np.full(alg.num_particles, -np.inf)

                    method = getattr(alg, alg_name.lower())
                    iterations, distance = method(src_position=np.zeros(self.dim))

                    fitness_values = np.array([func(pos.flatten()) for pos in alg.positions])
                    best_position = alg.positions[np.argmax(fitness_values)]
                    best_fitness = func(best_position.flatten())

                    # debug 
                    # print(f"{alg_name} on {func_name}: Best Position = {best_position}, Best Fitness = {best_fitness}")


                    global_minimum = func(np.zeros(self.dim))
                    if abs(best_fitness - global_minimum) < self.threshold:
                        success_count += 1

                    results[alg_name][func_name]["iterations"].append(iterations)
                    results[alg_name][func_name]["distance"].append(distance)

                results[alg_name][func_name]["success_rate"] = success_count / self.num_simulations

        return results

    def plot_results(self, results):
        metrics = ["iterations", "distance"]#, "success_rate"]

        for metric in metrics:
            plt.figure(figsize=(12, 8))
            for alg_name in self.algorithms.keys():
                averages = [np.mean(results[alg_name][func_name][metric]) if metric != "success_rate"
                            else results[alg_name][func_name][metric]
                            for func_name in self.benchmark_functions.keys()]
                plt.plot(self.benchmark_functions.keys(), averages, label=alg_name, marker='o')

            plt.title(f"Performance Comparison - {metric.capitalize()}", fontsize=16)
            plt.xlabel("Benchmark Function", fontsize=12)
            plt.ylabel(f"Average {metric.capitalize()}", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.xticks(rotation=45)
            # plt.savefig(f"./figure/Paper1/r1_with_noise/{metric}_bench.png", dpi = 300)
            plt.show()
            plt.close()


benchmark_functions = {
    "Sphere": sphere,
    "Schwefel 2.22": schwefel_222,
    "Schwefel 1.2": schwefel_12,
    "Quartic with Noise": quartic_with_noise,
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Griewank": griewank,
    "Schwefel": schwefel,
    "Hybrid 1": hybrid_1,
    "Hybrid 2": hybrid_2,
    "Composition 1": composition_1,
    "Composition 2": composition_2,
    "Rosenbrock": rosenbrock,
    "Schwefel 2.6": schwefel_26,
    "High Conditioned Elliptic": high_conditioned_elliptic
}


apso = APSO()
spso = SPSO()
arpso = ARPSO()

benchmark_performance = BenchmarkPerformance(
    algorithms={"APSO": apso, "SPSO": spso, "ARPSO": arpso},
    benchmark_functions=benchmark_functions,
    num_simulations=10,
    dim=2,
    threshold=0.1
)

results = benchmark_performance.evaluate()

benchmark_performance.plot_results(results)

# for func_name, func in benchmark_functions.items():
#     test_input = np.random.uniform(-100, 100, size=(2,))
#     try:
#         output = func(test_input)
#         print(f"Function: {func_name}, Input: {test_input}, Output: {output}, Output Shape: {np.shape(output)}")
#     except Exception as e:
#         print(f"Function: {func_name} raised an error: {e}")
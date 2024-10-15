import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def knapsack_rhc(weights, values, max_weight, max_attempts=100, max_iters=1000, restarts=10):

    total_weight = sum(weights)
    max_weight_pct = max_weight / total_weight

    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

    best_state, best_fitness, _ = mlrose.random_hill_climb(
        problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        restarts=restarts
    )

    return best_state, best_fitness

# Example usage:
weights = [2, 3, 4, 5]  # Item weights
values = [3, 4, 5, 6]   # Item values
max_weight = 5          # Maximum capacity of the knapsack

best_state, best_fitness = knapsack_rhc(weights, values, max_weight)

print("Best state (items selected):", best_state)
print("Best fitness (maximum value):", best_fitness)


def knapsack_sa(weights, values, max_weight, max_attempts=100, max_iters=1000, schedule=mlrose.GeomDecay()):

    total_weight = sum(weights)
    max_weight_pct = max_weight / total_weight

    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

    best_state, best_fitness, _ = mlrose.simulated_annealing(
        problem,
        schedule=schedule,
        max_attempts=max_attempts,
        max_iters=max_iters
    )

    return best_state, best_fitness

# Example usage:
weights = [2, 3, 4, 5]  # Item weights
values = [3, 4, 5, 6]   # Item values
max_weight = 5          # Maximum capacity of the knapsack

best_state, best_fitness = knapsack_sa(weights, values, max_weight)

print("Best state (items selected):", best_state)
print("Best fitness (maximum value):", best_fitness)

import mlrose_hiive as mlrose

def knapsack_ga(weights, values, max_weight, max_attempts=100, max_iters=1000, pop_size=200, mutation_prob=0.1):

   
    total_weight = sum(weights)
    max_weight_pct = max_weight / total_weight

    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

    best_state, best_fitness, _ = mlrose.genetic_alg(
        problem,
        pop_size=pop_size,
        mutation_prob=mutation_prob,
        max_attempts=max_attempts,
        max_iters=max_iters
    )

    return best_state, best_fitness

# Example usage:
weights = [2, 3, 4, 5]  # Item weights
values = [3, 4, 5, 6]   # Item values
max_weight = 5          # Maximum capacity of the knapsack

best_state, best_fitness = knapsack_ga(weights, values, max_weight)

print("Best state (items selected):", best_state)
print("Best fitness (maximum value):", best_fitness)

def knapsack_mimic(weights, values, max_weight, max_attempts=100, max_iters=1000, pop_size=200, keep_pct=0.2):

    total_weight = sum(weights)
    max_weight_pct = max_weight / total_weight

    fitness = mlrose.Knapsack(weights, values, max_weight_pct)

    problem = mlrose.DiscreteOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

    best_state, best_fitness, _ = mlrose.mimic(
        problem,
        pop_size=pop_size,
        keep_pct=keep_pct,
        max_attempts=max_attempts,
        max_iters=max_iters
    )

    return best_state, best_fitness

# Example usage:
weights = [2, 3, 4, 5]  # Item weights
values = [3, 4, 5, 6]   # Item values
max_weight = 5          # Maximum capacity of the knapsack

best_state, best_fitness = knapsack_mimic(weights, values, max_weight)

print("Best state (items selected):", best_state)
print("Best fitness (maximum value):", best_fitness)


def execute_randomized_algo(min_size=10, max_size=100, step=10, max_attempts=10, max_iters_multiplier=10, restart_multiplier=10, pop_size_multiplier=10, mutation_prob=0.4, keep_pct=0.2):
    
    results = {"RHC": [], "GA": [], "SA": [], "MIMIC": []}

    for size in range(min_size, max_size, step):
        print(f"Solving for problem size: {size}")

        weights = np.random.uniform(low=0.1, high=1, size=size)
        values = np.random.uniform(low=1, high=size, size=size)

        fitness = mlrose.Knapsack(weights, values)
        problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2)

        start_time = time.time()
        _, _, curve = mlrose.random_hill_climb(problem, restarts=restart_multiplier*size, max_attempts=max_attempts, max_iters=size*max_iters_multiplier, curve=True)
        end_time = time.time()
        results["RHC"].append(end_time - start_time)

        start_time = time.time()
        _, _, curve = mlrose.genetic_alg(problem, pop_size=pop_size_multiplier*size, mutation_prob=mutation_prob, max_attempts=max_attempts, max_iters=size*max_iters_multiplier, curve=True)
        end_time = time.time()
        results["GA"].append(end_time - start_time)

        start_time = time.time()
        _, _, curve = mlrose.simulated_annealing(problem, schedule=mlrose.GeomDecay(), max_attempts=max_attempts, max_iters=size*max_iters_multiplier, curve=True)
        end_time = time.time()
        results["SA"].append(end_time - start_time)

        start_time = time.time()
        _, _, curve = mlrose.mimic(problem, pop_size=pop_size_multiplier*size, keep_pct=keep_pct, max_attempts=max_attempts, max_iters=size*max_iters_multiplier, curve=True)
        end_time = time.time()
        results["MIMIC"].append(end_time - start_time)

    df = pd.DataFrame(results, index=range(min_size, max_size, step))

    # Plot the results
    ax = df.plot(marker='o', figsize=(12, 6))
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time Taken by Optimization Algorithms vs Problem Size")
    plt.show()

execute_randomized_algo(min_size=10, max_size=100, step=10)


import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def solve_n_queens_rhc(n, max_attempts=100, max_iters=1000, restarts=10):
    

    # Define the fitness function for the N-Queens problem
    fitness = mlrose.Queens()
    
    # Define the optimization problem object
    problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=False, max_val=n)
    
    # Solve the problem using Randomized Hill Climbing
    best_state, best_fitness, _ = mlrose.random_hill_climb(
        problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        restarts=restarts,
        curve=False
    )
    
    return best_state, best_fitness

# Example usage for N=8 (8-Queens problem)
n = 8
best_state, best_fitness = solve_n_queens_rhc(n)

print(f"Best state (positions of queens): {best_state}")
print(f"Best fitness (number of conflicting pairs): {best_fitness}")

import mlrose_hiive as mlrose
import numpy as np

def solve_n_queens_sa(n, max_attempts=100, max_iters=1000, schedule=mlrose.GeomDecay()):
    
    # Define the fitness function for the N-Queens problem
    fitness = mlrose.Queens()
    
    # Define the optimization problem object
    problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=False, max_val=n)
    
    # Solve the problem using Simulated Annealing
    best_state, best_fitness, _ = mlrose.simulated_annealing(
        problem,
        schedule=schedule,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=False
    )
    
    return best_state, best_fitness

# Example usage for N=8 (8-Queens problem)
n = 8
best_state, best_fitness = solve_n_queens_sa(n)

print(f"Best state (positions of queens): {best_state}")
print(f"Best fitness (number of conflicting pairs): {best_fitness}")


def solve_n_queens_ga(n, max_attempts=100, max_iters=1000, pop_size=200, mutation_prob=0.1):
    
    # Define the fitness function for the N-Queens problem
    fitness = mlrose.Queens()
    
    # Define the optimization problem object
    problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=False, max_val=n)
    
    # Solve the problem using Genetic Algorithm
    best_state, best_fitness, _ = mlrose.genetic_alg(
        problem,
        pop_size=pop_size,
        mutation_prob=mutation_prob,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=False
    )
    
    return best_state, best_fitness

# Example usage for N=8 (8-Queens problem)
n = 8
best_state, best_fitness = solve_n_queens_ga(n)

print(f"Best state (positions of queens): {best_state}")
print(f"Best fitness (number of conflicting pairs): {best_fitness}")


def solve_n_queens_mimic(n, max_attempts=100, max_iters=1000, pop_size=200, keep_pct=0.2):
    
    # Define the fitness function for the N-Queens problem
    fitness = mlrose.Queens()
    
    # Define the optimization problem object
    problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=False, max_val=n)
    
    # Solve the problem using MIMIC algorithm
    best_state, best_fitness, _ = mlrose.mimic(
        problem,
        pop_size=pop_size,
        keep_pct=keep_pct,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=False
    )
    
    return best_state, best_fitness

# Example usage for N=8 (8-Queens problem)
n = 8
best_state, best_fitness = solve_n_queens_mimic(n)

print(f"Best state (positions of queens): {best_state}")
print(f"Best fitness (number of conflicting pairs): {best_fitness}")



def compare_n_queens_algorithms(max_n=20, step=2, max_attempts=100, max_iters=1000, restarts=10, pop_size=200, mutation_prob=0.1, keep_pct=0.2):
    
    results = {
        "N": [],
        "RHC Time": [],
        "SA Time": [],
        "GA Time": [],
        "MIMIC Time": [],
        "RHC Fitness": [],
        "SA Fitness": [],
        "GA Fitness": [],
        "MIMIC Fitness": []
    }
    
    # Loop over problem sizes
    for n in range(4, max_n + 1, step):
        print(f"Solving N-Queens problem for N={n}")
        results["N"].append(n)
        
        # Define the fitness function for the N-Queens problem
        fitness = mlrose.Queens()
        problem = mlrose.DiscreteOpt(length=n, fitness_fn=fitness, maximize=False, max_val=n)
        
        # Randomized Hill Climbing (RHC)
        start_time = time.time()
        best_state, best_fitness, _ = mlrose.random_hill_climb(
            problem,
            max_attempts=max_attempts,
            max_iters=max_iters,
            restarts=restarts,
            curve=False
        )
        end_time = time.time()
        results["RHC Time"].append(end_time - start_time)
        results["RHC Fitness"].append(best_fitness)
        
        # Simulated Annealing (SA)
        start_time = time.time()
        best_state, best_fitness, _ = mlrose.simulated_annealing(
            problem,
            schedule=mlrose.GeomDecay(),
            max_attempts=max_attempts,
            max_iters=max_iters,
            curve=False
        )
        end_time = time.time()
        results["SA Time"].append(end_time - start_time)
        results["SA Fitness"].append(best_fitness)
        
        # Genetic Algorithm (GA)
        start_time = time.time()
        best_state, best_fitness, _ = mlrose.genetic_alg(
            problem,
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            max_iters=max_iters,
            curve=False
        )
        end_time = time.time()
        results["GA Time"].append(end_time - start_time)
        results["GA Fitness"].append(best_fitness)
        
        # MIMIC
        start_time = time.time()
        best_state, best_fitness, _ = mlrose.mimic(
            problem,
            pop_size=pop_size,
            keep_pct=keep_pct,
            max_attempts=max_attempts,
            max_iters=max_iters,
            curve=False
        )
        end_time = time.time()
        results["MIMIC Time"].append(end_time - start_time)
        results["MIMIC Fitness"].append(best_fitness)
    
    # Convert results to DataFrame for easy plotting
    df = pd.DataFrame(results)
    
    # Plot time comparison
    plt.figure(figsize=(12, 6))
    plt.plot(df["N"], df["RHC Time"], marker='o', label='RHC Time')
    plt.plot(df["N"], df["SA Time"], marker='o', label='SA Time')
    plt.plot(df["N"], df["GA Time"], marker='o', label='GA Time')
    plt.plot(df["N"], df["MIMIC Time"], marker='o', label='MIMIC Time')
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Time (seconds)")
    plt.title("Time Comparison for Solving N-Queens Problem")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot fitness comparison
    plt.figure(figsize=(12, 6))
    plt.plot(df["N"], df["RHC Fitness"], marker='o', label='RHC Fitness')
    plt.plot(df["N"], df["SA Fitness"], marker='o', label='SA Fitness')
    plt.plot(df["N"], df["GA Fitness"], marker='o', label='GA Fitness')
    plt.plot(df["N"], df["MIMIC Fitness"], marker='o', label='MIMIC Fitness')
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Fitness (Number of Conflicts)")
    plt.title("Fitness Comparison for Solving N-Queens Problem")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
compare_n_queens_algorithms(max_n=20, step=2)

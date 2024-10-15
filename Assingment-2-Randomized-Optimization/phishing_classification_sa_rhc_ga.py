import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlrose_hiive import NeuralNetwork, ExpDecay
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Initialize instances and load the dataset
def initialize_instances(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Assuming last column is the label
    y = data.iloc[:, -1].values
    return X, y

# Write results to a DataFrame using pd.concat for easier plotting
def append_results(results_df, algorithm_name, training_iterations, train_accuracy, training_time, test_accuracy, testing_time):
    new_row = pd.DataFrame([{
        "Algorithm": algorithm_name,
        "Iterations": training_iterations,
        "Training Accuracy": train_accuracy * 100,
        "Training Time (s)": training_time,
        "Testing Accuracy": test_accuracy * 100,
        "Testing Time (s)": testing_time
    }])
    return pd.concat([results_df, new_row], ignore_index=True)

# Training function
def train(model, X_train, y_train, iterations):
    model.set_params(max_iters=iterations)
    print(f"Starting training for {iterations} iterations.")
    model.fit(X_train, y_train)
    print(f"Completed training for {iterations} iterations.")

# Main function
def main(file_path):
    X, y = initialize_instances(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    hidden_layer = 50
    iterations = [10, 100, 500]

    # Define models for RHC, SA, and GA
    models = {
        "RHC": NeuralNetwork(
            hidden_nodes=[hidden_layer],
            activation='relu',
            algorithm='random_hill_climb',
            max_iters=5000,
            random_state=42,
            curve=True
        ),
        "SA": NeuralNetwork(
            hidden_nodes=[hidden_layer],
            activation='relu',
            algorithm='simulated_annealing',
            max_iters=5000,
            schedule=ExpDecay(init_temp=1e3, exp_const=0.4),
            random_state=42,
            curve=True
        ),
        "GA": NeuralNetwork(
            hidden_nodes=[hidden_layer],
            activation='relu',
            algorithm='genetic_alg',
            max_iters=5000,
            pop_size=75,
            mutation_prob=0.1,
            random_state=42,
            curve=True
        )
    }

    # DataFrame to store results
    results_df = pd.DataFrame(columns=[
        "Algorithm", "Iterations", "Training Accuracy", "Training Time (s)", "Testing Accuracy", "Testing Time (s)"
    ])

    for training_iterations in iterations:
        for name, model in models.items():
            start = time.time()
            train(model, X_train, y_train, training_iterations)
            training_time = time.time() - start

            # Training accuracy
            train_accuracy = accuracy_score(y_train, model.predict(X_train))

            # Testing accuracy
            start = time.time()
            test_accuracy = accuracy_score(y_test, model.predict(X_test))
            testing_time = time.time() - start

            # Append results to DataFrame
            results_df = append_results(
                results_df, name, training_iterations, train_accuracy, training_time, test_accuracy, testing_time
            )
            print(f"Completed {training_iterations} iterations for {name}.")

    # Plot the results
    plot_results(results_df)

# Function to plot the results
def plot_results(results_df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))

    # Plot Training Accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(data=results_df, x="Iterations", y="Training Accuracy", hue="Algorithm", marker='o')
    plt.title("Training Accuracy vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy (%)")
    plt.legend(title="Algorithm")

    # Plot Training Time
    plt.subplot(1, 2, 2)
    sns.lineplot(data=results_df, x="Iterations", y="Training Time (s)", hue="Algorithm", marker='o')
    plt.title("Training Time vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Training Time (seconds)")
    plt.legend(title="Algorithm")

    plt.tight_layout()
    plt.show()

    # Plot Testing Accuracy
    plt.figure(figsize=(7, 6))
    sns.lineplot(data=results_df, x="Iterations", y="Testing Accuracy", hue="Algorithm", marker='o')
    plt.title("Testing Accuracy vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Testing Accuracy (%)")
    plt.legend(title="Algorithm")
    plt.show()


if __name__ == "__main__":
    file_path = "phishing_web.csv"
    main(file_path)

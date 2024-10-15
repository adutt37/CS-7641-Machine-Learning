import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlrose_hiive import NeuralNetwork
import os
import csv
from datetime import datetime

# Initialize instances and load the dataset
def initialize_instances(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Assuming last column is the label
    y = data.iloc[:, -1].values
    return X, y

# Write results to a CSV file
def write_output_to_file(output_dir, file_name, results, final_result=False):
    if final_result:
        date_dir = os.path.join(output_dir, datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(date_dir, exist_ok=True)
        full_path = os.path.join(date_dir, file_name)
        with open(full_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results)
    else:
        full_path = os.path.join(output_dir, file_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)

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

    # Reduce iterations for quicker testing
    iterations = [10, 50, 100]
    population = [10, 25, 50]
    mate = [5, 10, 15]
    mutate = [2, 5, 10]

    results = []

    for training_iterations in iterations:
        for q in range(len(population)):
            start = time.time()
            ga_model = NeuralNetwork(
                hidden_nodes=[hidden_layer],
                activation='relu',
                algorithm='genetic_alg',
                max_iters=training_iterations,
                pop_size=population[q],
                mutation_prob=mutate[q] / 100,
                random_state=42,
                curve=True  # Add curve to track progress
            )

            train(ga_model, X_train, y_train, training_iterations)
            training_time = time.time() - start

            # Training accuracy
            train_accuracy = accuracy_score(y_train, ga_model.predict(X_train))

            # Testing accuracy
            start = time.time()
            test_accuracy = accuracy_score(y_test, ga_model.predict(X_test))
            testing_time = time.time() - start

            # Store results
            final_result = [
                "GA", training_iterations, population[q], mate[q], mutate[q],
                "training accuracy", f"{train_accuracy * 100:.2f}",
                "training time", f"{training_time:.2f}",
                "testing accuracy", f"{test_accuracy * 100:.2f}",
                "testing time", f"{testing_time:.2f}"
            ]
            write_output_to_file("Optimization_Results", "phishing_results_ga.csv", final_result, final_result=True)
            results.append(final_result)
            print(f"Completed {training_iterations} iterations with population {population[q]}.")

    for result in results:
        print(result)

# Run the main function with the file path
if __name__ == "__main__":
    file_path = "phishing_web.csv"
    main(file_path)

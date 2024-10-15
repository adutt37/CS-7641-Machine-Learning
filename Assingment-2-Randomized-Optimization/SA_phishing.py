import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlrose_hiive import NeuralNetwork, ExpDecay
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
def write_output_to_file(output_dir, file_name, results):
    date_dir = os.path.join(output_dir, datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(date_dir, exist_ok=True)
    full_path = os.path.join(date_dir, file_name)
    with open(full_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results)

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
    coolings = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    temperatures = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

    results = []

    for training_iterations in iterations:
        for cooling, temp in zip(coolings, temperatures):
            schedule = ExpDecay(init_temp=temp, exp_const=cooling, min_temp=1e-10)
            
            start = time.time()
            sa_model = NeuralNetwork(
                hidden_nodes=[hidden_layer],
                activation='relu',
                algorithm='simulated_annealing',
                max_iters=training_iterations,
                schedule=schedule,
                random_state=42,
                curve=True  # Track learning curve
            )

            train(sa_model, X_train, y_train, training_iterations)
            training_time = time.time() - start

            # Training accuracy
            train_accuracy = accuracy_score(y_train, sa_model.predict(X_train))

            # Testing accuracy
            start = time.time()
            test_accuracy = accuracy_score(y_test, sa_model.predict(X_test))
            testing_time = time.time() - start

            # Store results
            final_result = [
                "SA", training_iterations, cooling, temp,
                "training accuracy", f"{train_accuracy * 100:.2f}",
                "training time", f"{training_time:.2f}",
                "testing accuracy", f"{test_accuracy * 100:.2f}",
                "testing time", f"{testing_time:.2f}"
            ]
            write_output_to_file("Optimization_Results", "phishing_results_sa.csv", final_result)
            results.append(final_result)
            print(f"Completed {training_iterations} iterations with cooling {cooling} and temperature {temp}.")

    # Print all results at the end
    for result in results:
        print(result)


if __name__ == "__main__":
    file_path = "phishing_web.csv" 
    main(file_path)

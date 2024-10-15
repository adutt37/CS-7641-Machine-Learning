Overview
This repository contains implementations of optimization algorithms using the mlrose_hiive library. The code includes solutions for the Knapsack problem, N-Queens problem, and neural network training with various optimization strategies:

Randomized Hill Climbing (RHC)
Simulated Annealing (SA)
Genetic Algorithm (GA)
MIMIC (Mutual Information Maximizing Input Clustering)

Instructions to Execute the Python Files on your local machine:
1. First, clone the repository to your local machine
   git clone https://github.com/adutt37/CS-7641-Machine-Learning.git
   nagivate to folder Assingment-2-Randomized-Optimization

2. Install Dependencies.
    pip install mlrose-hiive numpy pandas matplotlib seaborn scikit-learn
   
4. Prepare the Datasets
Ensure that the datasets (phishing_web.csv) are present in the root directory of the project.

4. Execute the Python Scripts
To run the Python functions, follow these steps:

Run the Knapsack algorithms with the following command:
execute file py knapsack_random_search_algo.py

Run the N-queen algorithms with the following command:
execute file py nqueens_random_search_algo.py

Run the neural network algorithm with the following commands:
py GA_phishing.py
py RHC_phishing.py
py SA_phishing.py
phishing_classification_sa_rhc_ga.py






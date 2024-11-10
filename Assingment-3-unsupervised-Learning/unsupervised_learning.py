import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, homogeneity_score, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from collections import Counter
import timeit
import itertools
from sklearn.mixture import GaussianMixture
import timeit

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import SparseRandomProjection as RCA
from collections import defaultdict
from sklearn.metrics import f1_score


import matplotlib.pyplot as plt

def plot_model_training_times(num_samples, training_time_full, training_time_pca, training_time_ica, training_time_rca, title):
  
    plt.figure()
    plt.title(f"Training Times of Models: {title}")
    plt.xlabel("Sample Size")
    plt.ylabel("Training Time (s)")

    # Plotting each model's training times
    plt.plot(num_samples, training_time_full, linestyle='-', color="black", label="Full Model")
    plt.plot(num_samples, training_time_pca, linestyle='-', color="blue", label="PCA Model")
    plt.plot(num_samples, training_time_ica, linestyle='-', color="red", label="ICA Model")
    plt.plot(num_samples, training_time_rca, linestyle='-', color="green", label="RCA Model")

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


def plot_model_prediction_times(num_samples, pred_time_full, pred_time_pca, pred_time_ica, pred_time_rca, title):
   
    plt.figure()
    plt.title(f"Prediction Times of Models: {title}")
    plt.xlabel("Sample Size")
    plt.ylabel("Prediction Time (s)")

    # Plot each model’s prediction time
    plt.plot(num_samples, pred_time_full, linestyle='-', color="black", label="Full Model")
    plt.plot(num_samples, pred_time_pca, linestyle='-', color="blue", label="PCA Model")
    plt.plot(num_samples, pred_time_ica, linestyle='-', color="red", label="ICA Model")
    plt.plot(num_samples, pred_time_rca, linestyle='-', color="green", label="RCA Model")

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def plot_model_learning_rates(num_samples, f1_full, f1_pca, f1_ica, f1_rca, title):
    plt.figure()
    plt.title(f"Learning Rates of Models: {title}")
    plt.xlabel("Training Sample Size")
    plt.ylabel("F1 Score")

    # Plot the learning rates (F1 Scores) for each model
    plt.plot(num_samples, f1_full, linestyle='-', color="black", label="Full Dataset Model")
    plt.plot(num_samples, f1_pca, linestyle='-', color="blue", label="PCA Model")
    plt.plot(num_samples, f1_ica, linestyle='-', color="red", label="ICA Model")
    plt.plot(num_samples, f1_rca, linestyle='-', color="green", label="RCA Model")

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def load_and_preprocess_phishing_data(file_path):
    """
    Load and preprocess the phishing dataset from the specified file path.
    """
    # Read the dataset and convert all columns to category type
    phishing_data = pd.read_csv(file_path)
    phishing_data = phishing_data.astype('category')

    # Check for any missing values in the dataset
    if phishing_data.isnull().values.any():
        print("Alert: There are missing values in the data.")

    # Specify columns for one-hot encoding
    columns_to_encode = ['URL_Length', 'having_Sub_Domain', 'SSLfinal_State', 
                          'URL_of_Anchor', 'Links_in_tags', 'SFH', 
                          'web_traffic', 'Links_pointing_to_page']
    
    # Perform one-hot encoding on the specified columns
    encoded_df = pd.get_dummies(phishing_data[columns_to_encode])

    # Exclude the one-hot encoded columns and merge with the remaining data
    remaining_df = phishing_data.drop(columns=columns_to_encode)
    phishing_data = pd.concat([encoded_df, remaining_df], axis=1)

    # Replace instances of -1 with 0 and convert to category type
    phishing_data = phishing_data.replace(-1, 0).astype('category')

    # Reorder the DataFrame to move the target variable 'Result' to the first column
    columns = list(phishing_data)
    columns.insert(0, columns.pop(columns.index('Result')))
    phishing_data = phishing_data[columns]  # Reorganize columns

    # Split the DataFrame into features and target arrays
    features = np.array(phishing_data.iloc[:, 1:].values, dtype='int64')  # Exclude the target variable
    target = np.array(phishing_data.iloc[:, 0].values, dtype='int64')  # Target variable

    return features, target  # Return the features and target arrays


def load_preprocess_bank_data(file_path):
    """
    Load and preprocess the banking dataset.
    """
    # Load the banking dataset
    bank_data = pd.read_csv(file_path)
    print("Columns in the DataFrame:", bank_data.columns)

    # Separate features and target variable
    X = bank_data.drop('deposit', axis=1)  # Features without the target variable
    y = bank_data['deposit']  # Target variable

    # One-hot encode the features
    bank_data_encoded = pd.get_dummies(X, drop_first=True)

    # Prepare features and labels
    X = bank_data_encoded.values  # Convert to NumPy array after encoding
    y = y.map({'yes': 1, 'no': 0}).values  # Convert target variable to binary format

    return X, y  # Return features and target arrays


def perform_kmeans_clustering(features, true_labels, plot_title, max_k=50):
    """
    Executes K-means clustering on the specified dataset, evaluates results,
    and visualizes key metrics.

    Parameters:
    features (array-like): The feature set.
    true_labels (array-like): Actual labels of the data.
    plot_title (str): Title for generated plots.
    max_k (int): The upper limit for the number of clusters to explore.
    """
    k_values = list(np.arange(2, max_k, 2))
    silhouette_scores = []
    f1_scores = []
    homogeneity_scores = []
    training_durations = []

    for k in k_values:
        start_time = timeit.default_timer()
        kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=100)
        kmeans_model.fit(features)
        end_time = timeit.default_timer()

        training_durations.append(end_time - start_time)
        silhouette_scores.append(silhouette_score(features, kmeans_model.labels_))
        predicted_labels = predict_labels(true_labels, kmeans_model.labels_)
        f1_scores.append(f1_score(true_labels, predicted_labels))
        homogeneity_scores.append(homogeneity_score(true_labels, kmeans_model.labels_))

    # Visualize silhouette scores
    plt.figure()
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title(f'Silhouette Score Analysis: {plot_title}')
    plt.show()

    # Visualize homogeneity scores
    plt.figure()
    plt.plot(k_values, homogeneity_scores, marker='o')
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Homogeneity Score')
    plt.title(f'Homogeneity Scores: {plot_title}')
    plt.show()

    # Visualize training durations
    plt.figure()
    plt.plot(k_values, training_durations, marker='o')
    plt.grid(True)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Training Duration (seconds)')
    plt.title(f'K-Means Training Time: {plot_title}')
    plt.show()

    # Identify the optimal number of clusters
    best_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {best_k}")

    # Final evaluation with the best number of clusters
    final_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=100)
    final_kmeans.fit(features)
    assess_kmeans_model(final_kmeans, features, true_labels)

def predict_labels(true_labels, cluster_assignments):
    """
    Assigns predicted labels based on the most common label in each cluster.
    """
    assert (true_labels.shape == cluster_assignments.shape)
    predicted = np.empty_like(true_labels)

    for label in set(cluster_assignments):
        mask = cluster_assignments == label
        subset = true_labels[mask]
        majority_label = Counter(subset).most_common(1)[0][0]  # Most frequent label in the cluster
        predicted[mask] = majority_label

    return predicted

def assess_kmeans_model(kmeans_model, features, true_labels):
    """
    Evaluates the K-means clustering model based on the mode of predicted labels. """

    start_time = timeit.default_timer()
    kmeans_model.fit(features)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    predicted_labels = predict_labels(true_labels, kmeans_model.labels_)
    auc_score = roc_auc_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    confusion_mat = confusion_matrix(true_labels, predicted_labels)

    print("K-means Model Evaluation Metrics")
    print("*********************************")
    print(f"Training Duration (s): {training_time:.2f}")
    print(f"Iterations to Convergence: {kmeans_model.n_iter_}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}     AUC: {auc_score:.2f}")
    print(f"Precision: {precision:.2f}     Recall: {recall:.2f}")
    print("*********************************")
    
    plt.figure()
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])

    fmt = 'd'
    thresh = confusion_mat.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(confusion_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

X_phishing, y_phishing = load_and_preprocess_phishing_data('phishing_web.csv')
perform_kmeans_clustering(X_phishing, y_phishing, 'Phishing Data')

X_bank, Y_bank = load_preprocess_bank_data('bank.csv')
perform_kmeans_clustering(X_bank, Y_bank, 'Bank data')

#########################################################################################################################
##Expected Maximization

def access_em_model(em_model, features, true_labels):
  
    start_time = timeit.default_timer()
    em_model.fit(features)
    end_time = timeit.default_timer()
    training_duration = end_time - start_time
    
    predicted_labels = em_model.predict(features)
    majority_voted_labels = predict_labels(true_labels, predicted_labels)

    # Calculate evaluation metrics
    auc_score = roc_auc_score(true_labels, majority_voted_labels)
    f1_score_value = f1_score(true_labels, majority_voted_labels)
    accuracy_value = accuracy_score(true_labels, majority_voted_labels)
    precision_value = precision_score(true_labels, majority_voted_labels)
    recall_value = recall_score(true_labels, majority_voted_labels)
    confusion_mat = confusion_matrix(true_labels, majority_voted_labels)

    # Print evaluation metrics
    print("Model Evaluation Metrics Using Mode Cluster Vote")
    print("*****************************************************")
    print(f"Model Training Time (s):   {training_duration:.2f}")
    print(f"No. Iterations to Converge: {em_model.n_iter_}")
    print(f"Log-likelihood Lower Bound: {em_model.lower_bound_:.2f}")
    print(f"F1 Score:  {f1_score_value:.2f}")
    print(f"Accuracy:  {accuracy_value:.2f}     AUC: {auc_score:.2f}")
    print(f"Precision: {precision_value:.2f}     Recall: {recall_value:.2f}")
    print("*****************************************************")
    
    # Plot confusion matrix
    plt.figure()
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])

    fmt = 'd'
    thresh = confusion_mat.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(confusion_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def execute_expected_maximization(features, labels, plot_title):
    """
    Execute the Expectation-Maximization (EM) algorithm on the provided dataset.
    """
    num_components_range = list(np.arange(2, 100, 5))
    silhouette_scores = []
    f1_scores_list = []
    homogeneity_scores_list = []
    training_times = []
    aic_scores_list = []
    bic_scores_list = []
    
    # Initialize the variable to hold the last trained model
    last_em_model = None

    for num_components in num_components_range:
        start_time = timeit.default_timer()
        em_model = GaussianMixture(n_components=num_components, covariance_type='diag', n_init=1, warm_start=True, random_state=100)
        em_model.fit(features)
        end_time = timeit.default_timer()

        training_times.append(end_time - start_time)
        
        predicted_labels = em_model.predict(features)
        silhouette_scores.append(silhouette_score(features, predicted_labels))
        
        voted_labels = predict_labels(labels, predicted_labels)  # Using the same function to predict based on majority voting
        f1_scores_list.append(f1_score(labels, voted_labels))
        homogeneity_scores_list.append(homogeneity_score(labels, predicted_labels))
        
        aic_scores_list.append(em_model.aic(features))
        bic_scores_list.append(em_model.bic(features))

        # Store the last trained EM model
        last_em_model = em_model

    # Evaluate the last trained EM model
    access_em_model(last_em_model, features, labels)

    # Elbow curve for silhouette score
    plt.figure()
    plt.plot(num_components_range, silhouette_scores, marker='o')
    plt.grid(True)
    plt.xlabel('Number of Components')
    plt.ylabel('Average Silhouette Score')
    plt.title('Elbow Plot for EM: ' + plot_title)
    plt.show()

    # Plot homogeneity scores
    plt.figure()
    plt.plot(num_components_range, homogeneity_scores_list, marker='o')
    plt.grid(True)
    plt.xlabel('Number of Components')
    plt.ylabel('Homogeneity Score')
    plt.title('Homogeneity Scores EM: ' + plot_title)
    plt.show()

    # Plot F1 scores
    plt.figure()
    plt.plot(num_components_range, f1_scores_list, marker='o')
    plt.grid(True)
    plt.xlabel('Number of Components')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores EM: ' + plot_title)
    plt.show()

    # Plot model AIC and BIC
    plt.figure()
    plt.plot(num_components_range, aic_scores_list, label='AIC')
    plt.plot(num_components_range, bic_scores_list, label='BIC')
    plt.grid(True)
    plt.xlabel('Number of Components')
    plt.ylabel('Model Complexity Score')
    plt.title('EM Model Complexity: ' + plot_title)
    plt.legend(loc="best")
    plt.show()

# Usage Example:
X_phishing, y_phishing = load_and_preprocess_phishing_data('phishing_web.csv')
execute_expected_maximization(X_phishing, y_phishing, 'Phishing Data EM')

X_bank, Y_bank = load_preprocess_bank_data('bank.csv')
execute_expected_maximization(X_bank, Y_bank, 'Bank Data EM')

#####################################################################################

#Dimenstionality reduction- PCA
################################################################################
def perform_pca_analysis(data_features, data_labels, plot_title):
    
    # Display the first 10 labels for reference
    print("Sample Labels (first 10 entries):", data_labels[:10])  # Show initial labels

    # Initialize PCA and fit it to the features
    pca_instance = PCA(random_state=6).fit(data_features)
    cumulative_explained_variance = np.cumsum(pca_instance.explained_variance_ratio_)

    # Create a plot for cumulative explained variance
    figure, primary_axes = plt.subplots()
    primary_axes.plot(range(len(pca_instance.explained_variance_ratio_)), cumulative_explained_variance, 'b-')
    primary_axes.set_xlabel('Principal Components')
    primary_axes.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    primary_axes.tick_params(axis='y', labelcolor='b')
    plt.grid(False)

    # Create a secondary y-axis for eigenvalues
    secondary_axes = primary_axes.twinx()
    secondary_axes.plot(range(len(pca_instance.singular_values_)), pca_instance.singular_values_, 'm-')
    secondary_axes.set_ylabel('Eigenvalues', color='m')
    secondary_axes.tick_params(axis='y', labelcolor='m')
    plt.grid(False)

    # Set the title and display the plot
    plt.title("PCA Variance and Eigenvalues Visualization: " + plot_title)
    figure.tight_layout()
    plt.show()

# Usage Example:
X_phishing, y_phishing = load_and_preprocess_phishing_data('phishing_web.csv')
perform_pca_analysis(X_phishing, y_phishing, 'Phishing Data PCA')

X_bank, Y_bank = load_preprocess_bank_data('bank.csv')
perform_pca_analysis(X_bank, Y_bank, 'Bank Data PCA')

##########################################################
def perform_ica_analysis(features, labels, title):
    """
    Performs Independent Component Analysis (ICA) on the given features and visualizes kurtosis.

    Parameters:
    features (array-like): Feature set for ICA.
    labels (array-like): Corresponding labels (not used in ICA).
    title (str): Title for the plot.
    """
    # Define the range of dimensions for ICA
    dimensions = list(np.arange(2, (features.shape[1] - 1), 3))
    dimensions.append(features.shape[1])
    ica_model = ICA(random_state=5, max_iter=500, tol=0.001)  # Increased iterations and adjusted tolerance
    kurtosis_values = []

    # Compute kurtosis for different numbers of independent components
    for dim in dimensions:
        ica_model.set_params(n_components=dim)
        independent_components = ica_model.fit_transform(features)
        kurtosis = pd.DataFrame(independent_components).kurt(axis=0)
        kurtosis_values.append(kurtosis.abs().mean())

    # Plot the results
    plt.figure()
    plt.title("ICA Kurtosis: " + title)
    plt.xlabel("Independent Components")
    plt.ylabel("Average Kurtosis Across IC")
    plt.plot(dimensions, kurtosis_values, 'b-')
    plt.grid(False)
    plt.show()


X_phishing, y_phishing = load_and_preprocess_phishing_data('phishing_web.csv')
perform_ica_analysis(X_phishing, y_phishing, 'Phishing Data ICA')

X_bank, Y_bank = load_preprocess_bank_data('bank.csv')
perform_ica_analysis(X_bank, Y_bank, 'Bank Data ICA')
###############################################################################

def compute_pairwise_correlation(data1, data2):
    """
    Computes the correlation coefficient between the pairwise distances of two datasets.
    """
    # Ensure both datasets have the same number of samples
    assert data1.shape[0] == data2.shape[0], "Datasets must have the same number of samples."
    
    # Calculate pairwise distances for both datasets
    distances1 = pairwise_distances(data1)
    distances2 = pairwise_distances(data2)
    
    # Compute and return the correlation coefficient between the distance matrices
    correlation = np.corrcoef(distances1.ravel(), distances2.ravel())[0, 1]
    return correlation
############################################################################################
def run_random_projection_analysis(features, labels, title):
    """
    Performs random projection and visualizes mean and standard deviation of reconstruction correlation.
    """
    dimensions = list(np.arange(2, features.shape[1] - 1, 3))
    dimensions.append(features.shape[1])
    results = defaultdict(list)

    # Loop through dimensions and perform random projection
    for dim in dimensions:
        for i in range(5):  # Use 5 different random states
            rp_model = RCA(random_state=i, n_components=dim)
            projected_features = rp_model.fit_transform(features)
            # Store the correlation value
            correlation = compute_pairwise_correlation(projected_features, features)
            results[dim].append(correlation)

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results).T
    mean_reconstruction = results_df.mean(axis=1).tolist()
    std_reconstruction = results_df.std(axis=1).tolist()

    # Plotting
    fig, primary_axis = plt.subplots()
    primary_axis.plot(dimensions, mean_reconstruction, 'b-', label='Mean Reconstruction Correlation')
    primary_axis.set_xlabel('Random Components')
    primary_axis.set_ylabel('Mean Reconstruction Correlation', color='b')
    primary_axis.tick_params(axis='y', labelcolor='b')
    plt.grid(False)

    secondary_axis = primary_axis.twinx()
    secondary_axis.plot(dimensions, std_reconstruction, 'm-', label='STD Reconstruction Correlation')
    secondary_axis.set_ylabel('STD Reconstruction Correlation', color='m')
    secondary_axis.tick_params(axis='y', labelcolor='m')
    plt.grid(False)

    plt.title("Random Components for 5 Restarts: " + title)
    plt.legend(loc='upper left')
    fig.tight_layout()
    plt.show()


X_phishing, y_phishing = load_and_preprocess_phishing_data('phishing_web.csv')
run_random_projection_analysis(X_phishing, y_phishing, 'Phishing Data RCA')

X_bank, Y_bank = load_preprocess_bank_data('bank.csv')
run_random_projection_analysis(X_bank, Y_bank, 'Bank Data RCA')
##############################################################################################
def visualize_manifold_learning(features, labels, title):
    # Standardizing the features
    features_scaled = StandardScaler().fit_transform(features)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(features_scaled)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Classes')
    plt.show()

# Load and preprocess the datasets
X_phishing, y_phishing = load_and_preprocess_phishing_data('phishing_web.csv')
X_bank, y_bank = load_preprocess_bank_data('bank.csv')

# Visualize with t-SNE
visualize_manifold_learning(X_phishing, y_phishing, 't-SNE Visualization of Phishing Data')
visualize_manifold_learning(X_bank, y_bank, 't-SNE Visualization of Bank Data')
#######################################################################################################
## Re-apply clustering experiment on phising data
print("#######Re-apply clustering experiment on phising data#######")

X_phishing, y_phishing = load_and_preprocess_phishing_data('phishing_web.csv')
pca_data_for_phising = PCA(n_components=22,random_state=5).fit_transform(X_phishing)
ica_data_for_phising = ICA(n_components=38,random_state=5).fit_transform(X_phishing)
rca_data_for_phising = RCA(n_components=29,random_state=5).fit_transform(X_phishing)

perform_kmeans_clustering(pca_data_for_phising, y_phishing, 'PCA phising data')
perform_kmeans_clustering(ica_data_for_phising , y_phishing, 'ICA phising data')
perform_kmeans_clustering(rca_data_for_phising, y_phishing, 'RCA phising data')

execute_expected_maximization(pca_data_for_phising, y_phishing, 'PCA phising data EM')
execute_expected_maximization(ica_data_for_phising, y_phishing, 'ICA phising data EM')
execute_expected_maximization(rca_data_for_phising, y_phishing, 'RCA phising data EM')

print("#######Re-apply clustering experiment on bank data#######")
X_bank, y_bank = load_preprocess_bank_data('bank.csv')
pca_data_for_bank = PCA(n_components=22,random_state=5).fit_transform(X_bank)
ica_data_for_bank = ICA(n_components=38,random_state=5).fit_transform(X_bank)
rca_data_for_bank = RCA(n_components=29,random_state=5).fit_transform(X_bank)

perform_kmeans_clustering(pca_data_for_bank, y_bank, 'PCA bank data')
perform_kmeans_clustering(ica_data_for_bank, y_bank, 'ICA bank data')
perform_kmeans_clustering(rca_data_for_bank, y_bank, 'RCA bank data')

execute_expected_maximization(pca_data_for_bank, y_bank, 'PCA bank data EM')
execute_expected_maximization(ica_data_for_bank, y_bank, 'ICA bank data EM')
execute_expected_maximization(rca_data_for_bank, y_bank, 'RCA bank data EM')

################################################################################################################################

def plot_learning_curve(sample_sizes, mean_train_scores, std_train_scores, mean_cv_scores, std_cv_scores, graph_title):
  
    plt.figure()
    plt.title(f"Learning Curve: {graph_title}")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")

    # Plot training and cross-validation F1 scores with shaded regions for standard deviation
    plt.fill_between(sample_sizes, mean_train_scores - 2 * std_train_scores, mean_train_scores + 2 * std_train_scores, alpha=0.1, color="blue")
    plt.fill_between(sample_sizes, mean_cv_scores - 2 * std_cv_scores, mean_cv_scores + 2 * std_cv_scores, alpha=0.1, color="red")
    plt.plot(sample_sizes, mean_train_scores, 'o-', color="blue", label="Training Score")
    plt.plot(sample_sizes, mean_cv_scores, 'o-', color="red", label="Cross-Validation Score")
    
    plt.legend(loc="best")
    plt.show()

def plot_time_curve(sample_sizes, mean_fit_times, std_fit_times, mean_pred_times, std_pred_times, graph_title):
 
    plt.figure()
    plt.title(f"Modeling Time: {graph_title}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Time (s)")

    # Plot training and prediction times with shaded regions for standard deviation
    plt.fill_between(sample_sizes, mean_fit_times - 2 * std_fit_times, mean_fit_times + 2 * std_fit_times, alpha=0.1, color="blue")
    plt.fill_between(sample_sizes, mean_pred_times - 2 * std_pred_times, mean_pred_times + 2 * std_pred_times, alpha=0.1, color="red")
    plt.plot(sample_sizes, mean_fit_times, 'o-', color="blue", label="Training Time")
    plt.plot(sample_sizes, mean_pred_times, 'o-', color="red", label="Prediction Time")
    
    plt.legend(loc="best")
    plt.show()

def plot_performance_curves(model, features, labels, title="Insert Title"):
    
    n_samples = len(labels)
    train_sizes = (np.linspace(0.05, 1.0, 20) * n_samples).astype(int)

    mean_train_scores, std_train_scores = [], []
    mean_cv_scores, std_cv_scores = [], []
    mean_fit_times, std_fit_times = [], []
    mean_pred_times, std_pred_times = [], []

    for size in train_sizes:
        subset_indices = np.random.randint(features.shape[0], size=size)
        X_subset = features[subset_indices, :]
        y_subset = labels[subset_indices]
        
        # Cross-validation with F1 score and timing
        cv_results = cross_validate(
            model, X_subset, y_subset, cv=10, scoring='f1', n_jobs=-1, return_train_score=True
        )
        
        mean_train_scores.append(np.mean(cv_results['train_score']))
        std_train_scores.append(np.std(cv_results['train_score']))
        mean_cv_scores.append(np.mean(cv_results['test_score']))
        std_cv_scores.append(np.std(cv_results['test_score']))
        
        mean_fit_times.append(np.mean(cv_results['fit_time']))
        std_fit_times.append(np.std(cv_results['fit_time']))
        mean_pred_times.append(np.mean(cv_results['score_time']))
        std_pred_times.append(np.std(cv_results['score_time']))

    # Convert lists to arrays for plotting
    mean_train_scores = np.array(mean_train_scores)
    std_train_scores = np.array(std_train_scores)
    mean_cv_scores = np.array(mean_cv_scores)
    std_cv_scores = np.array(std_cv_scores)
    mean_fit_times = np.array(mean_fit_times)
    std_fit_times = np.array(std_fit_times)
    mean_pred_times = np.array(mean_pred_times)
    std_pred_times = np.array(std_pred_times)

    # Plot the learning and time curves
    plot_learning_curve(train_sizes, mean_train_scores, std_train_scores, mean_cv_scores, std_cv_scores, title)
    plot_time_curve(train_sizes, mean_fit_times, std_fit_times, mean_pred_times, std_pred_times, title)
    
    return train_sizes, mean_train_scores, mean_fit_times, mean_pred_times

def add_cluster_labels(X,km_lables,em_lables):
    
    df = pd.DataFrame(X)
    df['KM Cluster'] = km_lables
    df['EM Cluster'] = em_lables
    col_1hot = ['KM Cluster', 'EM Cluster']
    df_1hot = df[col_1hot]
    df_1hot = pd.get_dummies(df_1hot).astype('category')
    df_others = df.drop(col_1hot,axis=1)
    df = pd.concat([df_others,df_1hot],axis=1)
    new_X = np.array(df.values,dtype='int64')   
    
    return new_X



def evaluate_classifier_performance(model, X_train, X_test, y_train, y_test):
    # Measure training time
    start_time = timeit.default_timer()
    model.fit(X_train, y_train)
    training_duration = timeit.default_timer() - start_time

    # Measure prediction time
    start_time = timeit.default_timer()
    predictions = model.predict(X_test)
    prediction_duration = timeit.default_timer() - start_time

    # Compute evaluation metrics
    auc_score = roc_auc_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)

    # Display performance results
    print(f"Training Duration (s):   {training_duration:.5f}")
    print(f"Prediction Duration (s): {prediction_duration:.5f}\n")
    print(f"F1 Score:    {f1:.2f}")
    print(f"Accuracy:    {accuracy:.2f}     AUC:   {auc_score:.2f}")
    print(f"Precision:   {precision:.2f}     Recall: {recall:.2f}")
    print("*****************************************************")
    
    # Placeholder for potential confusion matrix or additional plots
    plt.figure()
    plt.title("Classifier Evaluation Plot")
    plt.show()

def execute_clustering_and_classification(data_features, data_labels, analysis_title):
    """
    Conduct clustering, dimensionality reduction, and classification on the dataset.
    """
    # KMeans Clustering
    kmeans_model = KMeans(n_clusters=9, n_init=10, random_state=100).fit(data_features)
    kmeans_labels = kmeans_model.labels_

    # Gaussian Mixture Model
    gmm_model = GaussianMixture(n_components=24, covariance_type='diag', n_init=1, warm_start=True, random_state=100).fit(data_features)
    gmm_labels = gmm_model.predict(data_features)

    # Dimensionality Reduction
    data_pca = PCA(n_components=22, random_state=5).fit_transform(data_features)
    data_ica = ICA(n_components=38, random_state=5).fit_transform(data_features)
    data_rca = RCA(n_components=29, random_state=5).fit_transform(data_features)

    # Combine clusters with original data
    combined_full = add_cluster_labels(data_features, kmeans_labels, gmm_labels)
    combined_pca = add_cluster_labels(data_pca, kmeans_labels, gmm_labels)
    combined_ica = add_cluster_labels(data_ica, kmeans_labels, gmm_labels)
    combined_rca = add_cluster_labels(data_rca, kmeans_labels, gmm_labels)

    # Train and evaluate on full dataset with clusters
    X_train, X_test, y_train, y_test = train_test_split(np.array(combined_full), np.array(data_labels), test_size=0.20)
    model_full = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=100)
    train_samples_full, nn_train_score_full, nn_fit_time_full, nn_pred_time_full = plot_performance_curves(model_full, X_train, y_train, title="Neural Net with Clusters: Full")
    evaluate_classifier_performance(model_full, X_train, X_test, y_train, y_test)

    # Train and evaluate on PCA-reduced dataset with clusters
    X_train, X_test, y_train, y_test = train_test_split(np.array(combined_pca), np.array(data_labels), test_size=0.20)
    model_pca = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=100)
    train_samples_pca, nn_train_score_pca, nn_fit_time_pca, nn_pred_time_pca = plot_performance_curves(model_pca, X_train, y_train, title="Neural Net with Clusters: PCA")
    evaluate_classifier_performance(model_pca, X_train, X_test, y_train, y_test)

    # Train and evaluate on ICA-reduced dataset with clusters
    X_train, X_test, y_train, y_test = train_test_split(np.array(combined_ica), np.array(data_labels), test_size=0.20)
    model_ica = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=100)
    train_samples_ica, nn_train_score_ica, nn_fit_time_ica, nn_pred_time_ica = plot_performance_curves(model_ica, X_train, y_train, title="Neural Net with Clusters: ICA")
    evaluate_classifier_performance(model_ica, X_train, X_test, y_train, y_test)

    # Train and evaluate on RCA-reduced dataset with clusters
    X_train, X_test, y_train, y_test = train_test_split(np.array(combined_rca), np.array(data_labels), test_size=0.20)
    model_rca = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=100)
    train_samples_rca, nn_train_score_rca, nn_fit_time_rca, nn_pred_time_rca = plot_performance_curves(model_rca, X_train, y_train, title="Neural Net with Clusters: RCA")
    evaluate_classifier_performance(model_rca, X_train, X_test, y_train, y_test)

    # Plotting performance metrics across all configurations
    plot_model_training_times(train_samples_full, nn_fit_time_full, nn_fit_time_pca, nn_fit_time_ica, nn_fit_time_rca, 'Dataset Analysis')
    plot_model_prediction_times(train_samples_full, nn_pred_time_full, nn_pred_time_pca, nn_pred_time_ica, nn_pred_time_rca, 'Dataset Analysis')
    plot_model_learning_rates(train_samples_full, nn_train_score_full, nn_train_score_pca, nn_train_score_ica, nn_train_score_rca, 'Dataset Analysis')

    

# Example usage
X_phishing, y_phishing = load_and_preprocess_phishing_data('phishing_web.csv')
execute_clustering_and_classification(X_phishing, y_phishing, 'Phishing Data Analysis')




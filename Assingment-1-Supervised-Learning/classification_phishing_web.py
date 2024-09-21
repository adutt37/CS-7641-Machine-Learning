import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


## Neural network train
def MlpClassifier_Model_Train(file_path, test_size=0.5, hidden_layer_sizes=(16, 8), max_iter=300):
   
 
    df = pd.read_csv(file_path)

    X = df.drop('Result', axis=1)
    y = df['Result']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

  
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=1)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred))

    plt.plot(clf.loss_curve_)
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


MlpClassifier_Model_Train('phishing_web.csv', test_size=0.5, hidden_layer_sizes=(16, 8), max_iter=300)


## Support Vector Machine 
def SVMKernel(file_path, test_size=0.5, C=1.0):
   
    df = pd.read_csv(file_path)

    X = df.drop('Result', axis=1)
    y = df['Result']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    results = []

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        if kernel == 'poly':
           
            for degree in range(2, 6): 
                svc = SVC(kernel=kernel, degree=degree, C=C, random_state=1)
                svc.fit(X_train, y_train)
                y_pred = svc.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results.append((f'{kernel} Degree {degree}', accuracy))
        else:
            svc = SVC(kernel=kernel, C=C, random_state=1)
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((kernel, accuracy))

   
    kernel_names = [k[0] for k in results]
    accuracies = [a[1] for a in results]

    plt.figure(figsize=(10, 5))
    plt.bar(kernel_names, accuracies, color='dodgerblue')
    plt.xlabel('Kernel Type')
    plt.ylabel('Accuracy')
    plt.title('Comparison of SVM Kernels')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

SVMKernel('phishing_web.csv', test_size=0.5, C=1.0)

##KNN Classifier

def KNNClassifier(file_path, test_size=0.5, neighbors_range=range(1, 16)):
    
  
    df = pd.read_csv(file_path)
    
    X = df.drop('Result', axis=1)
    y = df['Result']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    accuracies = []

    for n_neighbors in neighbors_range:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Number of Neighbors: {n_neighbors} - Test Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(neighbors_range, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.title('KNN Classifier Accuracy by Number of Neighbors')
    plt.grid(True)
    plt.show()


KNNClassifier('phishing_web.csv', test_size=0.5, neighbors_range=range(1, 16))

##GradientBoostingClassifier
def GradientBoostingClassifierModel(file_path, test_size=0.5, n_estimators=100, learning_rate=0.1, max_depth=3):
   
    # Load the data
    df = pd.read_csv(file_path)

    # Assuming the target variable is 'Result' and all other columns are features
    X = df.drop('Result', axis=1)
    y = df['Result']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Initialize the Gradient Boosting classifier
    gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

    # Fit the model
    gb.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = gb.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred))

    # Optional: Feature Importance Plot
    feature_importances = gb.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title('Feature Importances in Gradient Boosting Model')
    plt.bar(range(X_train.shape[1]), feature_importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [df.columns[i] for i in indices], rotation=90)
    plt.ylabel('Relative Importance')
    plt.show()

# Example of using the function
GradientBoostingClassifierModel('phishing_web.csv', test_size=0.5, n_estimators=100, learning_rate=0.1, max_depth=3)


#Compare all the model

def compare_classifiers(file_path, test_size=0.5):
    df = pd.read_csv(file_path)

    X = df.drop('Result', axis=1)
    y = df['Result']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    classifiers = {
        'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=1),
        'SVM (Poly 3)': SVC(kernel='poly', degree=3, C=1.0, random_state=1),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, random_state=1),
        'KNN (5 neighbors)': KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'MLP Classifier': MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=1)
    }

    accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy

    # Plotting
    names = list(accuracies.keys())
    values = list(accuracies.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, values, color='dodgerblue')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Different Classifiers')
    plt.xticks(rotation=45)
    plt.ylim([min(values) - 0.05, 1])  
    plt.show()

# Example of using the function
compare_classifiers('phishing_web.csv', test_size=0.5)
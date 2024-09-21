import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def MLPClassifierModel(file_path, epochs=50, batch_size=5, test_size=0.5, random_state=42):

    df = pd.read_csv(file_path)
   
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])  # Male = 1, Female = 0

    le_parental_support = LabelEncoder()
    df['ParentalSupport'] = le_parental_support.fit_transform(df['ParentalSupport'])  # High = 0, Low = 1, Medium = 2

    df['FinalGradeClass'] = pd.cut(df['FinalGrade'], bins=[0, 69, 84, 100], labels=['Low', 'Medium', 'High'])

    X = df[['Gender', 'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities', 'ParentalSupport']]
    y = df['FinalGradeClass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(12, 8), max_iter=epochs, random_state=random_state)

    mlp.fit(X_train_scaled, y_train)

    y_pred = mlp.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'])
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])

    plt.figure(figsize=(8, 6))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - MLPClassifier')
    plt.show()


# Calling the modified function
MLPClassifierModel('student_performance.csv', epochs=50, batch_size=5)


#SVM Model
def SVMKernel(file_path, kernels=['linear', 'poly', 'rbf', 'sigmoid'], test_size=0.5, random_state=42):
    
    df = pd.read_csv(file_path)

   
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])  # Male = 1, Female = 0

    le_parental_support = LabelEncoder()
    df['ParentalSupport'] = le_parental_support.fit_transform(df['ParentalSupport'])  # High = 0, Low = 1, Medium = 2

    
    df['FinalGradeClass'] = pd.cut(df['FinalGrade'], bins=[0, 69, 84, 100], labels=['Low', 'Medium', 'High'])

    
    X = df[['Gender', 'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities', 'ParentalSupport']]
    y = df['FinalGradeClass']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    reports = {}
    confusion_matrices = {}

    
    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel...")

        
        svm = SVC(kernel=kernel)
        svm.fit(X_train_scaled, y_train)

        
        y_pred = svm.predict(X_test_scaled)

        
        report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], labels=['Low', 'Medium', 'High'], zero_division=0)
        reports[kernel] = report

        
        cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
        confusion_matrices[kernel] = cm

        
        print(f"\nClassification Report for {kernel} kernel:\n", report)

    plt.figure(figsize=(16, 12))
    for i, kernel in enumerate(kernels):
        plt.subplot(2, 2, i + 1)
        sb.heatmap(confusion_matrices[kernel], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
        plt.title(f'Confusion Matrix - {kernel} kernel')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

    return reports, confusion_matrices


# Call the modified function
file_path = 'student_performance.csv'
reports, confusion_matrices = SVMKernel(file_path, kernels=['linear', 'poly', 'rbf', 'sigmoid'])

# Display classification reports for each kernel
for kernel, report in reports.items():
    print(f"SVM {kernel.capitalize()} Kernel Report:\n{report}\n")



def KNNClassifier(file_path, test_size=0.5, max_neighbors=10):
 
    df = pd.read_csv(file_path)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['ParentalSupport'] = le.fit_transform(df['ParentalSupport'])
    
    X = df[['Gender', 'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities', 'ParentalSupport']]
    y = df['FinalGrade']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=0)

    max_neighbors = min(max_neighbors, len(X_train) - 1) if len(X_train) > 1 else 1

    accuracies = []

    neighbors_settings = range(1, max_neighbors + 1)
    for n_neighbors in neighbors_settings:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors_settings, accuracies, marker='o')
    plt.title('KNN Accuracy vs. Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(min(neighbors_settings), max(neighbors_settings)+1, 1.0))
    plt.grid(True)
    plt.show()

# Example of using the function
KNNClassifier('student_performance.csv', test_size=0.5, max_neighbors=10)




def GradientBoostingClassifier_model(file_path, test_size=0.5, n_estimators=100, learning_rate=0.1, max_depth=3):
 
    df = pd.read_csv(file_path)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['ParentalSupport'] = le.fit_transform(df['ParentalSupport'])

    
    X = df[['Gender', 'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities', 'ParentalSupport']]
    y = df['FinalGrade']

   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

   
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=0)

   
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0)
    
   
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)

   
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

 
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), df.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

GradientBoostingClassifier_model('student_performance.csv', test_size=0.5, n_estimators=100, learning_rate=0.1, max_depth=3)


#Compare all models
def compare_models(file_path, test_size=0.5, random_state=42):
    
    df = pd.read_csv(file_path)

    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['ParentalSupport'] = le.fit_transform(df['ParentalSupport'])
    df['FinalGradeClass'] = pd.cut(df['FinalGrade'], bins=[0, 69, 84, 100], labels=['Low', 'Medium', 'High'])

    
    X = df[['Gender', 'AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities', 'ParentalSupport']]
    y = le.fit_transform(df['FinalGradeClass'])

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

   
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    
    models = {
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'SVM Linear': SVC(kernel='linear'),
        'SVM Poly': SVC(kernel='poly'),
        'SVM RBF': SVC(kernel='rbf'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
        'MLP Classifier': MLPClassifier(hidden_layer_sizes=(12, 8), max_iter=100, random_state=random_state)
    }

   
    results = {}
    
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f'{name} Accuracy: {accuracy:.2f}')

    
    plt.figure(figsize=(12, 6))
    algorithms = list(results.keys())
    accuracies = list(results.values())
    plt.bar(algorithms, accuracies, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
    plt.title('Model Comparison on Student Performance Dataset')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

compare_models('student_performance.csv')


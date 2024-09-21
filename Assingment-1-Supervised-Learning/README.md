Machine Learning Model Comparisons
This project contains Python scripts to train and evaluate different machine learning models on two datasets: phishing_web.csv and student_performance.csv. The models implemented include:

Multilayer Perceptron (MLP) Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Gradient Boosting Classifier
In addition to individual model training, the scripts compare the performance of these models using accuracy and classification metrics.

Getting Started & Prerequisites
For testing on your own machine, you need only to install python 3.6 and the following packages:

pandas, numpy, scikit-learn, matplotlib, seaborn and tensorflow

Datasets
1. Phishing Web Dataset
This dataset is used to evaluate the models based on website phishing detection.

phishing_web.csv structure:
Feature columns: various numeric indicators for phishing detection.
Target column: Result (binary classification, 0 or 1).
2. Student Performance Dataset
This dataset evaluates student performance based on various demographic and behavioral factors.

student_performance.csv structure:
Columns:
Gender: Male/Female.
AttendanceRate: Attendance rate in percentage.
StudyHoursPerWeek: Number of study hours per week.
PreviousGrade: Grade received in the previous evaluation.
ExtracurricularActivities: Yes/No indicator.
ParentalSupport: High/Medium/Low level of support.
FinalGrade: Final numeric grade.
Target Column: FinalGradeClass (categorized as 'Low', 'Medium', 'High').
Functions Overview
1. MLP Classifier (Neural Network)
File: MlpClassifier_Model_Train(), MLPClassifierModel()
Description: Trains an MLP (Multilayer Perceptron) classifier using a dataset, evaluates accuracy, and generates a classification report and confusion matrix.

Example usage:
MlpClassifier_Model_Train('phishing_web.csv', test_size=0.5, hidden_layer_sizes=(16, 8), max_iter=300)

2. Support Vector Machine (SVM)
File: SVMKernel()
Description: Trains SVM models with different kernel types (linear, poly, rbf, sigmoid) and evaluates their accuracy. Generates classification reports and confusion matrices for each kernel.
Example Usage:

SVMKernel('phishing_web.csv', kernels=['linear', 'poly', 'rbf', 'sigmoid'], test_size=0.5)

3.K-Nearest Neighbors (KNN)
File: KNNClassifier()
Description: Trains a KNN model with varying numbers of neighbors, evaluates accuracy, and plots accuracy vs. the number of neighbors.
Example Usage:
KNNClassifier('phishing_web.csv', test_size=0.5, neighbors_range=range(1, 16))

4.Gradient Boosting Classifier
File: GradientBoostingClassifierModel(), GradientBoostingClassifier_model()
Description: Trains a Gradient Boosting Classifier and evaluates accuracy. Also provides feature importance visualization for the student_performance.csv dataset.

Example Usage:
GradientBoostingClassifierModel('phishing_web.csv', test_size=0.5)

5.Model Comparisons
File: compare_classifiers(), compare_models()
Description: Compares the performance (accuracy) of multiple classifiers (SVM, KNN, MLP, Gradient Boosting) on both datasets and generates bar plots for comparison.

Example Usage:
compare_classifiers('phishing_web.csv', test_size=0.5)

Instructions to Execute the Python Files
1. First, clone the repository to your local machine

2. Install Dependencies.

3. Prepare the Datasets
Ensure that the datasets (phishing_web.csv and student_performance.csv) are present in the root directory of the project.

4. Execute the Python Scripts

To run the Python functions, follow these steps:

Open a Terminal (or Command Prompt)
Navigate to the project directory where the Python files are located.
Run the Python file that contains the desired function(s).

Example usage:
python -c "from classification_phishing_web import MlpClassifier_Model_Train; MlpClassifier_Model_Train('phishing_web.csv', test_size=0.5, hidden_layer_sizes=(16, 8), max_iter=300)"

if you want to run the python file directly then use below command:
python classification_student_performace_model.py 
python classification_phishing_web.py

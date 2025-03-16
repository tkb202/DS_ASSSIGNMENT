import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pickle
import streamlit as st 

df_diabetes = pd.read_csv("./diabetes.csv")

# Display the first few rows of the dataset
df_diabetes.head()

missing_values_diabetes = df_diabetes.isnull().sum()

# Display missing values for each column
missing_values_diabetes

summary_stats_diabetes = df_diabetes.describe()

# Display summary statistics
summary_stats_diabetes

df_diabetes.hist(bins=10, figsize=(14, 12), layout=(3, 3), color='#3498db')
plt.tight_layout()
plt.show()

# Set up for correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix_diabetes = df_diabetes.corr()
sns.heatmap(corr_matrix_diabetes, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zeros with the median of each column
for col in columns_with_zeros:
    df_diabetes[col] = df_diabetes[col].replace(0, df_diabetes[col].median())

# Display the first few rows to check if the imputation worked
df_diabetes.head()


# Features to normalize
features_to_normalize = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected features
df_diabetes[features_to_normalize] = scaler.fit_transform(df_diabetes[features_to_normalize])

# Display the first few rows after normalization
df_diabetes.head()

correlation_matrix = df_diabetes.corr().abs()

# Select the upper triangle of the correlation matrix
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.85
high_correlation_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]

# Display highly correlated features (if any)
high_correlation_features


# Separate features and target variable
X = df_diabetes.drop('Outcome', axis=1)  # Features
y = df_diabetes['Outcome']               # Target

# Apply SelectKBest with Chi-Square
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X, y)

# Apply SelectKBest with Mutual Information
mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k='all')
mutual_info_selector.fit(X, y)

# Get feature scores for both methods
chi2_scores = chi2_selector.scores_
mutual_info_scores = mutual_info_selector.scores_

# Combine feature names and scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_scores,
    'Mutual Info Score': mutual_info_scores
})

feature_scores.sort_values(by='Mutual Info Score', ascending=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression and Random Forest models
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)

# Define hyperparameter grids for GridSearchCV
logistic_param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
random_forest_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}

# Perform GridSearchCV to tune hyperparameters for both models
logistic_grid_search = GridSearchCV(logistic_model, logistic_param_grid, cv=5, scoring='accuracy')
random_forest_grid_search = GridSearchCV(random_forest_model, random_forest_param_grid, cv=5, scoring='accuracy')

# Fit both models on the training set
logistic_grid_search.fit(X_train, y_train)
random_forest_grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV for each model
best_logistic_model = logistic_grid_search.best_estimator_
best_random_forest_model = random_forest_grid_search.best_estimator_

# Predict on the test set using the best models
logistic_preds = best_logistic_model.predict(X_test)
random_forest_preds = best_random_forest_model.predict(X_test)

# Calculate evaluation metrics
logistic_accuracy = accuracy_score(y_test, logistic_preds)
random_forest_accuracy = accuracy_score(y_test, random_forest_preds)

logistic_precision = precision_score(y_test, logistic_preds)
random_forest_precision = precision_score(y_test, random_forest_preds)

logistic_recall = recall_score(y_test, logistic_preds)
random_forest_recall = recall_score(y_test, random_forest_preds)

logistic_f1 = f1_score(y_test, logistic_preds)
random_forest_f1 = f1_score(y_test, random_forest_preds)

# Create a performance comparison table
performance_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [logistic_accuracy, random_forest_accuracy],
    'Precision': [logistic_precision, random_forest_precision],
    'Recall': [logistic_recall, random_forest_recall],
    'F1 Score': [logistic_f1, random_forest_f1]
})

# Display the performance comparison table
performance_comparison
logistic_cm = confusion_matrix(y_test, logistic_preds)
random_forest_cm = confusion_matrix(y_test, random_forest_preds)


# Define function to plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrix for each model
plot_confusion_matrix(logistic_cm, "Logistic Regression")
plot_confusion_matrix(random_forest_cm, "Random Forest")

reduced_random_forest_param_grid = {
    'n_estimators': [100, 200],  # Reduced number of estimators
    'max_depth': [10, 20],  # Narrowing down the depth range
    'min_samples_split': [2, 5],  # Fewer options for splitting
    'min_samples_leaf': [1, 2],  # Reduced options for leaf size
    'class_weight': ['balanced']  # Focus on 'balanced' class weights
}

# Perform GridSearchCV again with the reduced hyperparameter grid
random_forest_grid_search_reduced = GridSearchCV(RandomForestClassifier(random_state=42),
                                                 reduced_random_forest_param_grid,
                                                 cv=5, scoring='recall')

# Fit the model to the training set
random_forest_grid_search_reduced.fit(X_train, y_train)

# Get the best parameters and retrain the Random Forest model
best_random_forest_model_reduced = random_forest_grid_search_reduced.best_estimator_

# Predict on the test set using the tuned Random Forest model
random_forest_preds_reduced = best_random_forest_model_reduced.predict(X_test)

# Calculate the new confusion matrix for Random Forest
random_forest_cm_reduced = confusion_matrix(y_test, random_forest_preds_reduced)

# Plot the updated confusion matrix for Random Forest
plot_confusion_matrix(random_forest_cm_reduced, "Tuned Random Forest (Reduced Grid)")

with open('best_random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(best_random_forest_model_reduced, model_file)

print("Model saved as 'best_random_forest_model.pkl'")

with open('best_random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app layout
st.title("Disease Prediction App")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
glucose = st.number_input("Glucose Level", min_value=50.0, max_value=200.0, value=100.0)
blood_pressure = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=300, value=200)
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
alcohol = st.selectbox("Do you consume alcohol?", ["Yes", "No"])
family_history = st.selectbox("Family History of Disease?", ["Yes", "No"])

# Map categorical inputs to numerical format
smoking = 1 if smoking == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
family_history = 1 if family_history == "Yes" else 0

# Prepare input features for the model
input_features = np.array([[age, bmi, glucose, blood_pressure, cholesterol, smoking, alcohol, family_history]])

# When the button is pressed, predict the result
if st.button("Predict"):
    prediction = model.predict(input_features)

    if prediction[0] == 1:
        st.write("Prediction: Disease Present")
    else:
        st.write("Prediction: No Disease")

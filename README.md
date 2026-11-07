
# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare five supervised machine learning classification algorithms to classify breast cancer samples into Benign and Malignant categories using the Breast Cancer Wisconsin Diagnostic dataset.

The project evaluates five classification models using six evaluation metrics and demonstrates their performance through an interactive Streamlit web application.

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin Diagnostic (WDBC)

**Source:** UCI Machine Learning Repository

**Dataset URL:**
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Dataset characteristics:

- Total records: 569
- Input features: 30
- Target variable: Diagnosis
- Classification type: Binary classification
- Benign samples: 357
- Malignant samples: 212
- Missing values: None

The dataset contains numerical measurements of cell nuclei obtained from digitized images of breast mass samples.

## 3. GitHub Repository Link

[ADD YOUR GITHUB REPOSITORY URL]

The repository contains the complete source code, training notebook, dataset, test data, trained classification models, application and dependencies.

## 4. Models Used

The following five algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Data Preprocessing

- Removed the ID column.
- Separated independent and dependent variables.
- Used an 80:20 train-test split.
- Applied stratified sampling.
- Applied StandardScaler through pipelines for Logistic Regression, KNN and SVM.

### Evaluation Metrics

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE |
| Decision Tree | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE |
| KNN | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE |
| Naive Bayes | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE |
| Support Vector Machine | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE | TO UPDATE |

### Model Performance Observations

**Logistic Regression:** To be updated based on actual experimental results.

**Decision Tree:** To be updated based on actual experimental results.

**KNN:** To be updated based on actual experimental results.

**Naive Bayes:** To be updated based on actual experimental results.

**Support Vector Machine:** To be updated based on actual experimental results.

### Overall Winner

To be identified based on the actual test results, considering all six evaluation metrics and malignant-class recall.

## 5. Streamlit Application

The interactive application includes:

- CSV test dataset upload.
- Classification model selection.
- Evaluation metrics display.
- Confusion matrix visualization.
- Classification report.
- Comparison of all five models.
- Downloadable comparison results.

The application loads previously trained models and evaluates uploaded test data without retraining.

## 6. Live Streamlit Application

[ADD YOUR STREAMLIT APPLICATION URL]

## 7. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 8. Conclusion

This project implements and compares five machine learning classification algorithms using the Breast Cancer Wisconsin Diagnostic dataset.

The models are evaluated using six performance metrics. An interactive Streamlit application demonstrates their performance on labelled test data.

The project covers the complete machine learning workflow, including dataset preparation, model training, evaluation, model saving and application deployment.

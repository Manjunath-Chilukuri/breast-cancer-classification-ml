# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop, evaluate and compare five supervised machine learning classification algorithms for breast cancer classification using the Breast Cancer Wisconsin Diagnostic (WDBC) dataset.

The project classifies samples into two categories: Benign and Malignant.

Five machine learning algorithms are implemented and evaluated using six performance metrics. An interactive Streamlit application is also developed to demonstrate model evaluation and comparison using labelled test data.

## 2. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin Diagnostic (WDBC)

**Dataset Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Dataset Characteristics:**

- Total instances: 569
- Input features: 30
- Target variable: Diagnosis
- Classification type: Binary Classification
- Benign samples: 357
- Malignant samples: 212
- Missing values: None

The dataset contains numerical measurements extracted from digitized images of fine needle aspirate (FNA) samples of breast masses.

The features describe characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity and symmetry.

## 3. GitHub Repository

**Repository Name:** breast-cancer-classification-ml

**GitHub URL:** https://github.com/Manjunath-Chilukuri/breast-cancer-classification-ml

The repository contains the source code, dataset, Jupyter Notebook, trained machine learning models, test dataset, Streamlit application and project documentation.

## 4. Machine Learning Algorithms Implemented

The following five supervised classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 5. Evaluation Metrics

All five classification algorithms were evaluated using six performance metrics.

**1. Accuracy:** Measures the proportion of correctly classified instances.

**2. AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between the two classes.

**3. Precision:** Measures the proportion of predicted positive instances that are actually positive.

**4. Recall:** Measures the proportion of actual positive instances correctly identified.

**5. F1 Score:** Represents the harmonic mean of precision and recall.

**6. Matthews Correlation Coefficient (MCC):** Measures the overall quality of binary classification predictions, considering all four confusion matrix outcomes.

## 6. Model Performance Comparison

The following table presents the actual experimental results obtained from the labelled test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 96.49% | 99.60% | 97.50% | 92.86% | 95.12% | 0.9245 |
| Decision Tree | 92.98% | 92.46% | 90.48% | 90.48% | 90.48% | 0.8492 |
| KNN | 95.61% | 98.23% | 97.44% | 90.48% | 93.83% | 0.9058 |
| Naive Bayes | 93.86% | 99.34% | 100.00% | 83.33% | 90.91% | 0.8715 |
| Support Vector Machine | 97.37% | 99.47% | 100.00% | 92.86% | 96.30% | 0.9442 |

## 7. Model-Wise Observations

### 7.1 Logistic Regression

- Accuracy: 96.49%
- AUC: 99.60%
- Precision: 97.50%
- Recall: 92.86%
- F1 Score: 95.12%
- MCC: 0.9245

**Observation:** Logistic Regression demonstrated excellent classification performance and achieved the highest AUC among all five models. It showed a strong ability to distinguish between benign and malignant samples.

### 7.2 Decision Tree Classifier

- Accuracy: 92.98%
- AUC: 92.46%
- Precision: 90.48%
- Recall: 90.48%
- F1 Score: 90.48%
- MCC: 0.8492

**Observation:** Decision Tree achieved balanced precision and recall. However, its overall accuracy, AUC and F1 score were comparatively lower than those of the other models.

### 7.3 K-Nearest Neighbors (KNN)

- Accuracy: 95.61%
- AUC: 98.23%
- Precision: 97.44%
- Recall: 90.48%
- F1 Score: 93.83%
- MCC: 0.9058

**Observation:** KNN achieved strong classification performance, with high precision and AUC. Its recall was slightly lower than Logistic Regression and SVM.

### 7.4 Gaussian Naive Bayes

- Accuracy: 93.86%
- AUC: 99.34%
- Precision: 100.00%
- Recall: 83.33%
- F1 Score: 90.91%
- MCC: 0.8715

**Observation:** Naive Bayes achieved perfect precision and a high AUC. However, it recorded the lowest recall among the five algorithms, indicating that it missed a comparatively higher proportion of actual positive-class instances.

### 7.5 Support Vector Machine (SVM)

- Accuracy: 97.37%
- AUC: 99.47%
- Precision: 100.00%
- Recall: 92.86%
- F1 Score: 96.30%
- MCC: 0.9442

**Observation:** SVM achieved the highest accuracy, F1 score and MCC among the five algorithms. It also achieved 100% precision, tied with Naive Bayes. Its high AUC and recall indicate strong classification performance.

## 8. Overall Best-Performing Model

**Selected Model: Support Vector Machine (SVM)**

Based on the experimental comparison, Support Vector Machine was identified as the best overall-performing classification algorithm.

Reasons for selecting SVM:

- Highest accuracy of 97.37%.
- Joint highest precision of 100%, shared with Naive Bayes.
- Highest F1 score of 96.30%.
- Highest MCC of 0.9442.
- Strong AUC of 99.47%.
- Recall of 92.86%.

Although Logistic Regression achieved the highest AUC of 99.60%, SVM demonstrated the strongest overall balance across the six evaluation metrics.

Therefore, SVM was selected as the overall best-performing model for this experiment.

## 9. Streamlit Application

An interactive web application was developed using Streamlit to demonstrate the machine learning model evaluation process.

**Application Features:**

- CSV test dataset upload.
- Dataset preview.
- Five machine learning model selection.
- Individual model evaluation.
- Accuracy, AUC, precision, recall, F1 score and MCC display.
- Confusion matrix visualization.
- Classification report.
- Five-model performance comparison.
- Downloadable model comparison results.

The application loads previously trained models and evaluates uploaded labelled test data.

**Live Application URL:** https://breast-cancer-classification-ml-3m3fz2qy3lzbfwrwdwkmtg.streamlit.app/

## 10. Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 11. Conclusion

This project successfully implemented and compared five supervised machine learning classification algorithms using the Breast Cancer Wisconsin Diagnostic dataset.

The experimental results demonstrated that all five models achieved promising classification performance, with Support Vector Machine providing the strongest overall results.

The project also developed an interactive Streamlit application to demonstrate model selection, evaluation metrics, confusion matrices, classification reports and model comparison.

This assignment provided practical experience in dataset preparation, supervised machine learning, model evaluation, performance comparison, model persistence and web application deployment.

**Note:** This project is intended for academic and educational purposes and is not a substitute for professional medical diagnosis.

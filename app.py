
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Breast Cancer ML Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #17365D;
}
.subtitle {
    color: #64748b;
    font-size: 15px;
}
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #17365D;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🔬 Breast Cancer Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Model Evaluation and Comparison Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "This academic application evaluates five trained "
    "classification algorithms using labelled test data. "
    "It is not intended for clinical diagnosis."
)

st.divider()

# =====================================================
# 4. MODEL FILE CONFIGURATION
# =====================================================

MODEL_PATHS = {
    "Logistic Regression":
        "model/logistic_regression.joblib",

    "Decision Tree":
        "model/decision_tree.joblib",

    "K-Nearest Neighbors":
        "model/knn.joblib",

    "Gaussian Naive Bayes":
        "model/naive_bayes.joblib",

    "Support Vector Machine":
        "model/svm.joblib"
}


# =====================================================
# 5. LOAD SAVED MODELS
# =====================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, path in MODEL_PATHS.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded_models[name] = joblib.load(path)

    return loaded_models


@st.cache_resource
def load_feature_names():

    path = "model/feature_names.joblib"

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Feature names file not found."
        )

    return joblib.load(path)


try:

    models = load_models()

    feature_names = load_feature_names()

except Exception as error:

    st.error(f"Model loading failed: {error}")

    st.stop()


# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("Project Information")

    st.write("**Project:** Breast Cancer Classification")

    st.write("**Dataset:** WDBC")

    st.write("**Input Features:** 30")

    st.write("**Classification:** Binary")

    st.write("**Algorithms:** 5")

    st.divider()

    st.subheader("Available Models")

    for name in models.keys():
        st.write("✓", name)

    st.divider()

    st.caption(
        "BITS Pilani | M.Tech AIML | Machine Learning Assignment 2"
    )


# =====================================================
# 7. DATASET UPLOAD
# =====================================================

st.markdown(
    '<div class="section-title">1. Upload Test Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Select your labelled test CSV file",
    type=["csv"],
    help="Upload test_data.csv containing all 30 features and Diagnosis."
)

if uploaded_file is None:

    st.info(
        "Please upload your test_data.csv file to begin evaluation."
    )

    st.stop()


try:

    test_df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read CSV: {error}")

    st.stop()


# =====================================================
# 8. DATASET VALIDATION
# =====================================================

required_columns = feature_names + ["Diagnosis"]

missing_columns = [
    col for col in required_columns
    if col not in test_df.columns
]

if missing_columns:

    st.error(
        f"Missing required columns: {missing_columns}"
    )

    st.stop()


if test_df.empty:

    st.error("The uploaded dataset is empty.")

    st.stop()


if test_df[required_columns].isnull().any().any():

    st.error(
        "The uploaded dataset contains missing values."
    )

    st.stop()


X_uploaded = test_df[feature_names].copy()

y_actual = test_df["Diagnosis"].copy()


try:

    X_uploaded = X_uploaded.apply(
        pd.to_numeric,
        errors="raise"
    )

except Exception:

    st.error(
        "All 30 input features must be numerical."
    )

    st.stop()


if not np.isfinite(X_uploaded.to_numpy()).all():

    st.error(
        "The dataset contains infinite or invalid numerical values."
    )

    st.stop()


if not set(y_actual.unique()).issubset({"B", "M"}):

    st.error(
        "Diagnosis must contain only B and M."
    )

    st.stop()


if set(y_actual.unique()) != {"B", "M"}:

    st.error(
        "Both B and M classes must be present for AUC evaluation."
    )

    st.stop()


# =====================================================
# 9. DATASET INFORMATION
# =====================================================

st.success("Dataset uploaded and validated successfully!")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Test Records",
    len(test_df)
)

col2.metric(
    "Input Features",
    len(feature_names)
)

col3.metric(
    "Target Classes",
    2
)

with st.expander("Preview Uploaded Dataset"):

    st.dataframe(
        test_df.head(10),
        use_container_width=True
    )


# =====================================================
# 10. MODEL SELECTION
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">2. Select Classification Model</div>',
    unsafe_allow_html=True
)

selected_model_name = st.selectbox(
    "Choose an algorithm",
    list(models.keys())
)

selected_model = models[selected_model_name]


# =====================================================
# 11. EVALUATION FUNCTION
# =====================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    positive_index = list(
        model.classes_
    ).index("M")

    probabilities = model.predict_proba(X)[
        :, positive_index
    ]

    actual_binary = (y == "M").astype(int)

    metrics = {

        "Accuracy": accuracy_score(
            y, predictions
        ),

        "AUC": roc_auc_score(
            actual_binary, probabilities
        ),

        "Precision": precision_score(
            y, predictions,
            pos_label="M",
            zero_division=0
        ),

        "Recall": recall_score(
            y, predictions,
            pos_label="M",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y, predictions,
            pos_label="M",
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y, predictions
        )
    }

    return metrics, predictions


# =====================================================
# 12. EVALUATE SELECTED MODEL
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">3. Model Evaluation</div>',
    unsafe_allow_html=True
)

if st.button(
    "Evaluate Selected Model",
    type="primary",
    use_container_width=True
):

    metrics, predictions = evaluate_model(
        selected_model,
        X_uploaded,
        y_actual
    )

    st.session_state["metrics"] = metrics

    st.session_state["predictions"] = predictions

    st.session_state["model_name"] = selected_model_name


# =====================================================
# 13. DISPLAY SIX METRICS
# =====================================================

if (
    "metrics" in st.session_state
    and st.session_state["model_name"] == selected_model_name
):

    metrics = st.session_state["metrics"]

    predictions = st.session_state["predictions"]

    st.subheader(
        f"Performance: {selected_model_name}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")

    c2.metric("AUC Score", f"{metrics['AUC']:.4f}")

    c3.metric("Precision", f"{metrics['Precision']:.4f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("Recall", f"{metrics['Recall']:.4f}")

    c5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

    c6.metric("MCC Score", f"{metrics['MCC']:.4f}")


    # =================================================
    # 14. CONFUSION MATRIX
    # =================================================

    st.divider()

    st.subheader("4. Confusion Matrix")

    cm = confusion_matrix(
        y_actual,
        predictions,
        labels=["B", "M"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax
    )

    ax.set_xlabel("Predicted Diagnosis")

    ax.set_ylabel("Actual Diagnosis")

    ax.set_title(
        f"Confusion Matrix - {selected_model_name}"
    )

    st.pyplot(fig)

    plt.close(fig)


    # =================================================
    # 15. CLASSIFICATION REPORT
    # =================================================

    st.divider()

    st.subheader("5. Classification Report")

    report = classification_report(
        y_actual,
        predictions,
        labels=["B", "M"],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


# =====================================================
# 16. COMPARE ALL FIVE MODELS
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">6. Compare All Five Models</div>',
    unsafe_allow_html=True
)

if st.button(
    "Run Five-Model Comparison",
    use_container_width=True
):

    comparison_results = []

    for name, model in models.items():

        metrics, predictions = evaluate_model(
            model,
            X_uploaded,
            y_actual
        )

        comparison_results.append({
            "ML Model": name,
            **metrics
        })

    comparison_df = pd.DataFrame(
        comparison_results
    )

    st.session_state["comparison"] = comparison_df


if "comparison" in st.session_state:

    comparison_df = st.session_state["comparison"]

    st.subheader("Model Performance Comparison")

    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True
    )

    st.subheader("Metric Comparison Chart")

    chart_df = comparison_df.set_index(
        "ML Model"
    )[
        ["Accuracy", "AUC", "F1 Score", "MCC"]
    ]

    st.bar_chart(chart_df)

    st.download_button(
        "Download Comparison Results",
        data=comparison_df.to_csv(
            index=False
        ),
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )


# =====================================================
# 17. FOOTER
# =====================================================

st.divider()

st.caption(
    "Breast Cancer Wisconsin Diagnostic Dataset | "
    "BITS Pilani M.Tech AIML | Machine Learning Assignment 2"
)

"""
Student Depression Prediction and Exploratory Data Analysis (EDA)
=================================================================
Course: Introduction to Data Science, 2nd Year 1st Term - De La Salle Lipa
Contributors: Irish May Pureza, Julia Lois Mendoza, Kent

This script performs:
1. Data Ingestion & Integrity Checks
2. Comprehensive Univariate Data Exploration (Target, Numerical, Categorical)
3. Bivariate Data Exploration (Features vs Depression)
4. Numerical and Categorical Correlation / Association Analysis
5. Preprocessing & Feature Encoding (Ordinal, Nominal, Boolean mappings)
6. Feature Scaling & Train-Validation Splitting
7. Logistic Regression Modeling, Local Validation Dashboard & Confusion Matrix
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set aesthetic style for visualizations
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "figure.autolayout": True,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


# ==========================================
# 1. DATA INGESTION & INSPECTION
# ==========================================
def load_data(file_candidates=None):
    """Load dataset from potential paths (local or Google Colab)."""
    if file_candidates is None:
        file_candidates = [
            "Student Depression Dataset.csv.xlsx",
            "/content/Student Depression Dataset.csv.xlsx",
            Path(__file__).parent / "Student Depression Dataset.csv.xlsx",
        ]

    for path in file_candidates:
        if os.path.exists(path):
            print(f"[INFO] Successfully loaded dataset from: {path}")
            return pd.read_excel(path)

    raise FileNotFoundError(
        "Could not locate 'Student Depression Dataset.csv.xlsx'. "
        "Please ensure the file is in the project directory."
    )


def summarize_dataset(df: pd.DataFrame):
    """Print structural and statistical summaries of the dataset."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Total Duplicates: {df.duplicated().sum()}")
    print("\nColumn Information:")
    print(df.info())
    print("\nDescriptive Statistics (Numerical):")
    print(df.describe().T)
    print("=" * 60)


# ==========================================
# 2. UNIVARIATE EXPLORATORY DATA ANALYSIS
# ==========================================
def perform_univariate_eda(df: pd.DataFrame, output_dir: str = "figures"):
    """Generate and save univariate distributions for target, numeric, and categorical variables."""
    os.makedirs(output_dir, exist_ok=True)
    print("[EDA] Generating Univariate Visualizations...")

    # A. Target Variable Distribution (Depression)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    target_counts = df["Depression"].value_counts()
    target_labels = ["Depressed (1)", "Not Depressed (0)"]
    colors = ["#e74c3c", "#3498db"]

    sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
        palette=colors,
        ax=axes[0],
        hue=target_counts.index,
        legend=False,
    )
    axes[0].set_title("Depression Class Frequency")
    axes[0].set_xlabel("Depression Status")
    axes[0].set_ylabel("Student Count")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Not Depressed (0)", "Depressed (1)"])
    for i, v in enumerate(target_counts.values):
        axes[0].text(
            i,
            v + 300,
            f"{v:,} ({v / len(df) * 100:.1f}%)",
            ha="center",
            fontweight="bold",
        )

    axes[1].pie(
        target_counts,
        labels=target_labels,
        autopct="%1.2f%%",
        startangle=140,
        colors=colors,
        explode=(0.04, 0),
        textprops={"fontweight": "bold"},
    )
    axes[1].set_title("Depression Class Ratio")
    plt.suptitle("Target Distribution: Student Depression", fontsize=16)
    plt.savefig(
        os.path.join(output_dir, "01_target_distribution.png"), dpi=300
    )
    plt.close()

    # B. Numerical Variables Distribution
    numeric_cols = [
        "Age",
        "Academic Pressure",
        "Work Pressure",
        "CGPA",
        "Study Satisfaction",
        "Job Satisfaction",
        "Work/Study Hours",
        "Financial Stress",
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        sns.histplot(
            df[col].dropna(),
            kde=True,
            ax=axes[idx],
            color="#2980b9",
            bins=20 if df[col].nunique() > 10 else df[col].nunique(),
        )
        axes[idx].set_title(f"Distribution of {col}", fontweight="bold")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Count")

    plt.suptitle("Univariate Analysis: Numerical Feature Distributions", y=1.01)
    plt.savefig(
        os.path.join(output_dir, "02_univariate_numerical.png"), dpi=300
    )
    plt.close()

    # C. Categorical Variables Distribution
    cat_cols = [
        "Gender",
        "Sleep Duration",
        "Dietary Habits",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
        "Profession",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    axes = axes.flatten()

    for idx, col in enumerate(cat_cols):
        top_cats = df[col].value_counts().head(8)
        sns.barplot(
            y=top_cats.index,
            x=top_cats.values,
            ax=axes[idx],
            palette="viridis",
            hue=top_cats.index,
            legend=False,
        )
        axes[idx].set_title(f"Distribution of {col}", fontweight="bold")
        axes[idx].set_xlabel("Count")
        axes[idx].set_ylabel(col)
        for i, v in enumerate(top_cats.values):
            axes[idx].text(v + 100, i, f"{v:,}", va="center", fontsize=9)

    plt.suptitle(
        "Univariate Analysis: Categorical Feature Distributions", y=1.01
    )
    plt.savefig(
        os.path.join(output_dir, "03_univariate_categorical.png"), dpi=300
    )
    plt.close()


# ==========================================
# 3. BIVARIATE EXPLORATORY DATA ANALYSIS
# ==========================================
def perform_bivariate_eda(df: pd.DataFrame, output_dir: str = "figures"):
    """Generate and save bivariate visualizations comparing predictors with Depression."""
    os.makedirs(output_dir, exist_ok=True)
    print("[EDA] Generating Bivariate Visualizations...")

    # A. Numerical Features vs Depression (Boxplots)
    key_numeric = [
        "Academic Pressure",
        "Financial Stress",
        "Work/Study Hours",
        "Study Satisfaction",
        "Age",
        "CGPA",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(key_numeric):
        sns.boxplot(
            x="Depression",
            y=col,
            data=df,
            palette=["#3498db", "#e74c3c"],
            ax=axes[idx],
            hue="Depression",
            legend=False,
        )
        axes[idx].set_title(f"{col} by Depression Status", fontweight="bold")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(["Not Depressed (0)", "Depressed (1)"])

    plt.suptitle(
        "Bivariate Analysis: Numerical Predictors vs Depression", y=1.01
    )
    plt.savefig(
        os.path.join(output_dir, "04_bivariate_numerical_vs_depression.png"),
        dpi=300,
    )
    plt.close()

    # B. Categorical Features vs Depression (Depression Rate by Category)
    key_cat = [
        "Have you ever had suicidal thoughts ?",
        "Dietary Habits",
        "Sleep Duration",
        "Family History of Mental Illness",
        "Gender",
        "Profession",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, col in enumerate(key_cat):
        # Calculate depression rate per category
        dep_rate = (
            df.groupby(col)["Depression"]
            .mean()
            .reset_index()
            .sort_values(by="Depression", ascending=False)
        )
        if col == "Profession":
            dep_rate = dep_rate.head(6)

        sns.barplot(
            data=dep_rate,
            x=col,
            y="Depression",
            palette="mako",
            ax=axes[idx],
            hue=col,
            legend=False,
        )
        axes[idx].set_title(
            f"Depression Rate by {col}", fontweight="bold", fontsize=12
        )
        axes[idx].set_ylabel("Depression Rate (0.0 to 1.0)")
        axes[idx].set_ylim(0, 1.0)
        axes[idx].tick_params(axis="x", rotation=25)

        for p in axes[idx].patches:
            height = p.get_height()
            axes[idx].annotate(
                f"{height * 100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height + 0.02),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle(
        "Bivariate Analysis: Categorical Features vs Depression Rate", y=1.01
    )
    plt.savefig(
        os.path.join(output_dir, "05_bivariate_categorical_vs_depression.png"),
        dpi=300,
    )
    plt.close()


# ==========================================
# 4. CORRELATION & ASSOCIATION ANALYSIS
# ==========================================
def cramers_v(x, y):
    """Compute bias-corrected Cramer's V for categorical association."""
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2corr / denom)


def perform_correlation_analysis(
    df: pd.DataFrame, X_encoded: pd.DataFrame, y: pd.Series, output_dir: str = "figures"
):
    """
    Perform both Numerical Correlation (Pearson) and Categorical Association (Cramer's V),
    along with global feature correlation ranking with the target.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("[EDA] Generating Correlation and Association Visualizations...")

    # A. Numerical Correlation Heatmap
    numeric_cols = [
        "Age",
        "Academic Pressure",
        "Work Pressure",
        "CGPA",
        "Study Satisfaction",
        "Job Satisfaction",
        "Work/Study Hours",
        "Financial Stress",
        "Depression",
    ]
    num_df = df[numeric_cols].copy()
    num_corr = num_df.corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        num_corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(
        "Numerical Features Correlation Matrix (Pearson)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.savefig(
        os.path.join(output_dir, "06_numerical_correlation_heatmap.png"), dpi=300
    )
    plt.close()

    # B. Categorical Association with Depression (Cramer's V)
    cat_cols = [
        "Have you ever had suicidal thoughts ?",
        "Dietary Habits",
        "Degree",
        "Sleep Duration",
        "City",
        "Family History of Mental Illness",
        "Profession",
        "Gender",
    ]
    cv_scores = {}
    for col in cat_cols:
        cv_scores[col] = cramers_v(df[col].astype(str), df["Depression"])

    cv_series = pd.Series(cv_scores).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        cv_series.index,
        cv_series.values,
        color=sns.color_palette("rocket", len(cv_series)),
    )
    plt.title(
        "Categorical Features Association with Depression (Cramér's V)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Cramér's V Statistic (0 = Independent, 1 = Perfect Association)")
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
    plt.xlim(0, 0.65)
    plt.savefig(
        os.path.join(output_dir, "07_categorical_cramers_v_heatmap.png"), dpi=300
    )
    plt.close()

    # C. Global Feature Correlation with Depression (Encoded Features)
    corr_with_y = X_encoded.corrwith(y).sort_values()

    # Filter top 10 positive and top 10 negative correlations
    top_corrs = pd.concat([corr_with_y.head(10), corr_with_y.tail(10)])

    plt.figure(figsize=(11, 8))
    colors = ["#e74c3c" if val > 0 else "#2ecc71" for val in top_corrs.values]
    bars = plt.barh(top_corrs.index, top_corrs.values, color=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(
        "Top Positive & Negative Feature Correlations with Depression",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Correlation Coefficient with Depression (y)")
    for bar in bars:
        width = bar.get_width()
        offset = 0.01 if width >= 0 else -0.04
        plt.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    plt.xlim(-0.35, 0.65)
    plt.savefig(
        os.path.join(output_dir, "08_all_features_depression_correlation.png"),
        dpi=300,
    )
    plt.close()


# ==========================================
# 5. DATA PREPROCESSING & FEATURE ENCODING
# ==========================================
def preprocess_dataset(df: pd.DataFrame):
    """
    Preprocess the dataset following the exact steps established by the team:
    - Split features (X) and target (y)
    - Median imputation for 'Financial Stress'
    - Irish May Pureza: One-Hot encode Gender, City, Profession; map Sleep Duration
    - Julia Lois Mendoza: Map Dietary Habits, LabelEncode Degree, map boolean indicators
    """
    print("[PREPROCESSING] Executing data transformations...")

    # Drop target and identifier
    X = df.drop(columns=["Depression", "id"]).copy()
    y = df["Depression"].copy()

    # Impute missing values in Financial Stress with median
    financial_median = X["Financial Stress"].median()
    X["Financial Stress"] = X["Financial Stress"].fillna(financial_median)
    print(f"  - Imputed Financial Stress missing values with median: {financial_median}")

    # --- Irish May Pureza's Section ---
    one_hot_cat_cols = ["Gender", "City", "Profession"]
    X = pd.get_dummies(X, columns=one_hot_cat_cols, drop_first=True)

    sleep_mapping = {
        "Less than 5 hours": 0,
        "5-6 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3,
    }
    X["Sleep Duration"] = X["Sleep Duration"].replace({"Others": "5-6 hours"})
    X["Sleep Duration"] = X["Sleep Duration"].str.strip().map(sleep_mapping)
    print("  - Completed Irish May Pureza's transformations: One-Hot & Sleep Duration mapped.")

    # --- Julia Lois Mendoza's Section ---
    X["Dietary Habits"] = X["Dietary Habits"].replace({"Others": "Healthy"})
    diet_mapping = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
    X["Dietary Habits"] = X["Dietary Habits"].map(diet_mapping)

    le = LabelEncoder()
    X["Degree"] = le.fit_transform(X["Degree"].astype(str))

    boolean_map = {"No": 0, "Yes": 1}
    bool_cols = [
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
    ]
    for col in bool_cols:
        X[col] = X[col].map(boolean_map)
    print("  - Completed Julia Lois Mendoza's transformations: Diet, Degree & Booleans mapped.")

    return X, y


# ==========================================
# 6. MODEL TRAINING & VALIDATION DASHBOARD
# ==========================================
def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series, output_dir: str = "figures"
):
    """
    Train Logistic Regression with standardized features and print validation dashboard.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[MODELING] Splitting dataset and scaling features...")

    # Train / Validation Split (80/20, random_state=84)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=84
    )

    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Model Training
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    val_preds = model.predict(X_val_scaled)

    # Metrics
    val_acc = accuracy_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)

    print("\n" + "=" * 40)
    print("===== LOCAL VALIDATION DASHBOARD =====")
    print("=" * 40)
    print(f"Accuracy            : {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Precision           : {val_precision:.4f} ({val_precision * 100:.2f}%)")
    print(f"Sensitivity (Recall): {val_recall:.4f} ({val_recall * 100:.2f}%)")
    print(f"F1-Score            : {val_f1:.4f}")
    print("=" * 40)

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_val, val_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Depressed (0)", "Depressed (1)"],
    )
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Confusion Matrix - Student Depression", fontsize=14, pad=12)
    plt.grid(False)
    plt.savefig(
        os.path.join(output_dir, "09_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"[INFO] Saved confusion matrix to {output_dir}/09_confusion_matrix.png")

    return model, {
        "accuracy": val_acc,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
    }


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    # 1. Load Data
    df = load_data()
    summarize_dataset(df)

    # 2. Univariate EDA
    perform_univariate_eda(df, output_dir="figures")

    # 3. Bivariate EDA
    perform_bivariate_eda(df, output_dir="figures")

    # 4. Preprocessing
    X, y = preprocess_dataset(df)

    # 5. Correlation Analysis (Numerical & Categorical)
    perform_correlation_analysis(df, X, y, output_dir="figures")

    # 6. Model Training & Evaluation
    train_and_evaluate(X, y, output_dir="figures")

    print("\n[SUCCESS] Entire data science pipeline executed successfully!")
    print("[INFO] All figures saved to the './figures' directory.")


if __name__ == "__main__":
    main()

# Student Depression Prediction and Exploratory Data Analysis (EDA)

**Course:** Introduction to Data Science, 2nd Year 1st Term — De La Salle Lipa  

---

## Project Overview

This repository contains the complete, cleaned, and properly formatted exploratory data analysis (EDA) and predictive modeling pipeline for the **Student Depression Dataset** (27,901 student records across 18 features).

The workflow investigates demographic, academic, financial, and behavioral factors associated with student depression and trains a **Logistic Regression** classifier.

### Local Validation Performance
| Metric | Value | Percentage | Benchmark Note |
| :--- | :---: | :---: | :--- |
| **Majority Baseline** | `0.5762` | **57.62%** | Naive majority class (1) guessing |
| **Model Accuracy** | `0.8556` | **85.56%** | **+27.94% absolute lift** over baseline |
| **Precision** | `0.8612` | **86.12%** | Reliability of positive depression flags |
| **Sensitivity (Recall)** | `0.8933` | **89.33%** | True positive cases detected (2,873 / 3,216) |
| **F1-Score** | `0.8770` | **—** | Robust harmonic mean across classes |

### Classification Outcomes Breakdown (Error Analysis)
| Outcome Category | Predicted Label | Actual Label | Count | Percentage | Operational Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **True Positives (TP)** | Depressed (1) | Depressed (1) | **2,873** | 51.48% | Vulnerable students correctly flagged |
| **True Negatives (TN)** | Not Depressed (0) | Not Depressed (0) | **1,902** | 34.08% | Non-depressed students correctly cleared |
| **False Positives (FP)** | Depressed (1) | Not Depressed (0) | **463** | 8.30% | False alarms requiring counseling review |
| **False Negatives (FN)** | Not Depressed (0) | Depressed (1) | **343** | 6.15% | Missed at-risk cases |
| **Total Validation** | — | — | **5,581** | 100.00% | 85.56% overall accuracy |

---

## Exploratory Data Analysis (EDA) Highlights

### 1. Univariate Analysis
- **Target Imbalance:** ~58.55% of students in the dataset report experiencing depression (`1`), while ~41.45% do not (`0`).
- **Academic & Financial Stress:** Both exhibit distributions heavily centered in the higher stress ratings (scale 3 to 5).
- **Sleep & Diet:** Over 70% of respondents sleep 6 hours or fewer per day; unhealthy/moderate diets dominate.

### 2. Bivariate Analysis & Depression Trends
- **Suicidal Thoughts:** Strongest categorical association with depression (88.7% depression rate among those answering "Yes" vs. 34.1% for "No").
- **Academic Pressure:** Depressed students report a median academic pressure rating of 4.0 compared to 2.0 for non-depressed students.
- **Financial Stress & Study Hours:** Both are substantially elevated among depressed students.

### 3. Correlation & Association Findings (Both Numerical & Categorical)
- **Top Risk Factors (Positive Correlation with Depression):**
  1. `Have you ever had suicidal thoughts ?` ($r \approx +0.546$, Cramér's V $\approx 0.546$)
  2. `Academic Pressure` ($r \approx +0.475$)
  3. `Financial Stress` ($r \approx +0.364$)
  4. `Work/Study Hours` ($r \approx +0.209$)
  5. `Family History of Mental Illness` ($r \approx +0.053$, Cramér's V $\approx 0.053$)
- **Top Protective Factors (Negative Correlation with Depression):**
  1. `Age` ($r \approx -0.226$) — younger students in this dataset report higher depression rates
  2. `Dietary Habits` ($r \approx -0.206$, Cramér's V $\approx 0.207$) — healthier diets correlate with lower depression rates
  3. `Study Satisfaction` ($r \approx -0.168$) — higher satisfaction is associated with lower depression
  4. `Sleep Duration` ($r \approx -0.087$, Cramér's V $\approx 0.099$) — longer sleep duration correlates with lower depression

---

## Project Structure

```text
student-depression/
├── Student Depression Dataset.csv.xlsx      # Original Excel dataset
├── student_depression_analysis.py          # Standalone executable pipeline script
├── student_depression_analysis.ipynb       # Formatted Jupyter notebook (Jupyter / Colab)
├── requirements.txt                        # Python dependencies
├── info.txt                                # Course information (DLSL)
├── README.md                               # Project documentation & summary
└── figures/                                # High-resolution EDA and evaluation plots
    ├── 01_target_distribution.png
    ├── 02_univariate_numerical.png
    ├── 03_univariate_categorical.png
    ├── 04_bivariate_numerical_vs_depression.png
    ├── 05_bivariate_categorical_vs_depression.png
    ├── 06_numerical_correlation_heatmap.png
    ├── 07_categorical_cramers_v_heatmap.png
    ├── 08_all_features_depression_correlation.png
    ├── 09_confusion_matrix.png
    ├── 10_logistic_regression_advantages.png
    ├── 11_logistic_regression_pipeline.png
    ├── 12_decision_threshold.png
    ├── 13_sigmoid_curve.png
    ├── 14_l2_regularization.png
    ├── 15_model_performance_table.png
    └── 16_classification_outcomes_table.png
```

---

## How to Run

### Option A: Running the Standalone Python Script
Using standard `pip`:
```bash
pip install -r requirements.txt
python student_depression_analysis.py
```

Using `uv`:
```bash
uv run --with-requirements requirements.txt python student_depression_analysis.py
```

### Option B: Running the Jupyter Notebook
Open `student_depression_analysis.ipynb` in:
- **VS Code** (built-in Jupyter extension)
- **JupyterLab / Notebook**: `jupyter notebook student_depression_analysis.ipynb`
- **Google Colab**: Upload the `.ipynb` and `Student Depression Dataset.csv.xlsx` to your Google Drive / Colab session runtime.

# Student Depression Prediction and Exploratory Data Analysis (EDA)

**Course:** Introduction to Data Science, 2nd Year 1st Term — De La Salle Lipa  
**Contributors:** Irish May Pureza, Julia Lois Mendoza, Kent  

---

## 📌 Project Overview

This repository contains the complete, cleaned, and properly formatted exploratory data analysis (EDA) and predictive modeling pipeline for the **Student Depression Dataset** (27,901 student records across 18 features).

The workflow investigates demographic, academic, financial, and behavioral factors associated with student depression and trains a **Logistic Regression** classifier.

### Local Validation Performance
| Metric | Value | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | `0.8556` | **85.56%** |
| **Precision** | `0.8612` | **86.12%** |
| **Sensitivity (Recall)** | `0.8933` | **89.33%** |
| **F1-Score** | `0.8770` | **—** |

---

## 📊 Exploratory Data Analysis (EDA) Highlights

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

## 📂 Project Structure

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
    └── 09_confusion_matrix.png
```

---

## 🚀 How to Run

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

---

## 👥 Team Work Breakdown

- **Irish May Pureza:**
  - One-hot encoding of nominal variables: `Gender`, `City`, `Profession` (`drop_first=True`).
  - Handling and ordinal mapping of `Sleep Duration` (`Less than 5 hours`: 0, `5-6 hours`: 1, `7-8 hours`: 2, `More than 8 hours`: 3, `'Others'` $\rightarrow$ `'5-6 hours'`).
- **Julia Lois Mendoza:**
  - Data cleaning and ordinal mapping for `Dietary Habits` (`Unhealthy`: 0, `Moderate`: 1, `Healthy`: 2, `'Others'` $\rightarrow$ `'Healthy'`).
  - Label encoding for `Degree`.
  - Binary mapping (`No`: 0, `Yes`: 1) for `Have you ever had suicidal thoughts ?` and `Family History of Mental Illness`.
- **Kent:**
  - Data ingestion, integrity & duplicate checking.
  - Univariate & bivariate visualization suite.
  - Pearson numerical correlation and Cramér's V categorical association analyses.
  - Train/Validation split (`80/20`, `random_state=84`), `StandardScaler`, and `LogisticRegression` baseline.
  - Validation metrics dashboard and Confusion Matrix reporting.

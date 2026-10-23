# AI Diabetes Prediction

A machine learning project for predicting diabetes risk using survey-based lifestyle and health indicators from the CDC's Behavioral Risk Factor Surveillance System (BRFSS), based on the `prosperchuks/health-dataset`.

The workflow spans exploratory data analysis (EDA), feature selection, data preprocessing, model experimentation (Random Forest, Hyperparameter Tuning via RandomizedSearchCV, and XGBoost), domain feature engineering, and decision threshold calibration optimized for clinical recall.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [2. Exploratory Data Analysis & Feature Selection](#2-exploratory-data-analysis--feature-selection)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training & Tuning](#4-model-training--tuning)
  - [5. Decision Threshold Calibration](#5-decision-threshold-calibration)
- [Model Performance Summary](#model-performance-summary)
- [Key Insights & Feature Importance](#key-insights--feature-importance)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)

---

## Overview

Diabetes is a significant chronic health condition worldwide. Early identification of individuals at risk enables timely medical intervention and lifestyle modification. This project develops and evaluates tree-based machine learning pipelines (Random Forest and XGBoost) to classify individuals as healthy (`0.0`) or diabetic (`1.0`) while prioritizing clinical recall to minimize false negatives.

---

## Dataset

- **Source:** Kaggle [`prosperchuks/health-dataset`](https://www.kaggle.com/datasets/prosperchuks/health-dataset) (`diabetes_data.csv`) downloaded automatically via `kagglehub`.
- **Size:** 70,692 observations, 18 initial columns.
- **Missing Values:** None (0 null values).
- **Class Balance:** Perfectly balanced target (`Diabetes`):
  - Healthy (`0.0`): 35,346 samples (50.0%)
  - Diabetes (`1.0`): 35,346 samples (50.0%)
- **Data Split:** 80% train (56,553 samples), 20% test (14,139 samples) with `random_state=42`.

---

## Project Workflow

### 1. Data Loading & Preprocessing
- Automated dataset fetch using `kagglehub.dataset_download`.
- Stratified/reproducible train-test split (`test_size=0.2`, `random_state=42`).
- Standardized continuous features (`BMI`) using `StandardScaler` fitted strictly on training data to prevent data leakage.

### 2. Exploratory Data Analysis & Feature Selection
- Correlation heatmaps and feature correlations against the target variable.
- Correlation filtering: retained features with absolute correlation \(|r| > 0.10\) against `Diabetes`.
- Analyzed distribution across key health indicators including `GenHlth`, `HighBP`, `BMI`, `Age`, and `HighChol`.

### 3. Feature Engineering
Domain-specific interaction terms and composite risk scores were engineered:
- **Cardiovascular Risk Score (`Cardio_Risk_Score`)**: Sum of co-morbid cardiovascular risk conditions (`HighBP + HighChol + HeartDiseaseorAttack + Stroke`).
- **BMI × Age Interaction (`BMI_x_Age`)**: Interaction capturing metabolic risk accumulation across aging cohorts.
- **Health Burden Index (`Health_Burden`)**: Interaction combining self-rated general health and physical health limitation days (`GenHlth * (PhysHlth + 1)`).

### 4. Model Training & Tuning
- **Random Forest Classifier**:
  - Baseline model evaluated with `n_estimators=100`, `max_depth=10`.
  - Feature ablation experiment tested top 8 vs. all 11 correlated features.
  - Hyperparameter optimization via `RandomizedSearchCV` (exploring depth, estimator counts, sample splits, and feature selection strategies).
- **XGBoost Classifier**:
  - Gradient-boosted trees configured with `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.
  - Retrained on engineered feature sets to capture non-linear relationships and interactions.

### 5. Decision Threshold Calibration
In clinical diagnostics, false negatives (failing to identify an individual with diabetes) carry a higher risk than false positives. The classification decision boundary was tuned:
- Default threshold: `0.50` (Recall: 0.80)
- Calibrated threshold: `0.45` (Recall boosted to **0.84** on diabetic class)

---

## Model Performance Summary

Evaluated on the held-out test set (\(N = 14,139\)):

| Model | Setup / Features | Test Accuracy | Diabetic Precision | Diabetic Recall | Diabetic F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | 11 correlated features (`max_depth=10`) | 74.91% | 0.73 | 0.79 | 0.76 |
| **Random Forest (Slim)** | Top 8 features by importance | 74.61% | — | — | — |
| **Random Forest (Tuned)** | `RandomizedSearchCV` (`max_depth=8`) | 74.77% | — | — | — |
| **XGBoost (Baseline)** | Correlated features | 75.10% | 0.73 | 0.80 | 0.76 |
| **XGBoost + Feature Engineering** | Engineered risk scores & interactions | 75.17% | 0.73 | 0.80 | 0.76 |
| **XGBoost (Threshold = 0.45)** | Engineered features, calibrated decision boundary | **75.00%** | **0.71** | **0.84** | **0.77** |

### Confusion Matrix (XGBoost @ Threshold 0.45)

```text
               Predicted Negative (0)    Predicted Positive (1)
Actual Healthy (0)           4,662                     2,428
Actual Diabetes (1)          1,123                     5,926
```

---

## Key Insights & Feature Importance

According to the XGBoost feature importance attribution:
1. **HighBP (High Blood Pressure)** (~40.3% relative importance) — strongest individual predictive indicator.
2. **Cardio_Risk_Score** (~21.8%) — composite cardiovascular comorbidity score created during feature engineering proved to be the second most influential predictor.
3. **GenHlth (General Health Rating)** (~16.0%) — strong patient self-assessment proxy for underlying chronic conditions.
4. **Age & Interaction Terms (`BMI_x_Age`, `Health_Burden`)** — meaningful secondary signals reflecting cumulative metabolic strain.

---

## Project Structure

```text
AI-diabetes/
├── 01_diabetes_ml.ipynb   # Main Jupyter notebook containing EDA, modeling, and evaluation
├── pyproject.toml         # Project metadata and dependencies (managed via uv)
├── uv.lock                # Locked dependency tree
├── README.md              # Project documentation
└── src/
    └── ai_diabetes/       # Source package directory
        └── __init__.py
```

---

## Installation & Setup

### Prerequisites
- Python `>= 3.12`
- Recommended: [`uv`](https://docs.astral.sh/uv/) (or standard `pip` / `venv`)

### Installation with `uv`

```bash
# Clone the repository
git clone <repo-url>
cd AI-diabetes

# Install dependencies into virtual environment
uv sync
```

### Installation with standard `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

---

## Usage

Launch Jupyter Lab or Notebook to run the pipeline:

```bash
uv run jupyter lab
```

Open [`01_diabetes_ml.ipynb`](01_diabetes_ml.ipynb) and run all cells. The notebook will automatically download the dataset via `kagglehub`, perform data preparation, train the models, and output performance metrics and visualizations.

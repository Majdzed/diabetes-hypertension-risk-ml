# Diabetes, Hypertension, and Stroke Prediction

End-to-end machine learning pipelines for clinical risk screening across three chronic conditions. This project focuses on **data integrity auditing**, **domain-specific feature engineering**, **class imbalance handling**, and **decision threshold optimization** to minimize false negatives in medical screening.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure & Results](#repository-structure--results)
- [Notebooks](#notebooks)
  - [01 — Diabetes](#01--diabetes)
  - [02 — Hypertension / CVD](#02--hypertension--cvd)
  - [03 — Stroke](#03--stroke)
- [Key Engineering Highlights](#key-engineering-highlights)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)

---

## Overview

Early identification of individuals at risk for chronic conditions enables timely medical intervention and lifestyle modification. This project develops and evaluates tree-based machine learning pipelines (Random Forest and XGBoost) across three datasets, each requiring its own data integrity audit, feature engineering strategy, and threshold calibration to minimize missed diagnoses (false negatives).

---

## Repository Structure & Results

| Notebook | Condition | Clean Dataset Size | Final Model | Decision Threshold | Key Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `01_diabetes_ml.ipynb` | **Diabetes** | 70,692 rows | `XGBClassifier` | `0.450` | **0.84 Recall** on positive cases |
| `02_hypertension_ml.ipynb` | **Hypertension / CVD** | 68,539 rows | `XGBClassifier` (`RandomizedSearchCV`) | `0.420` | **0.802 ROC-AUC** \| **0.76 Recall** \| **73.5% Acc** |
| `03_stroke_ml.ipynb` | **Stroke** | 3,425 unique rows | `RandomForest` (`class_weight='balanced'`) | `0.373` *(Youden's J)* | **0.72 Recall** on 18:1 imbalanced data |

---

## Notebooks

### 01 — Diabetes

**Dataset:** Kaggle [`prosperchuks/health-dataset`](https://www.kaggle.com/datasets/prosperchuks/health-dataset) (`diabetes_data.csv`) — 70,692 rows, 18 columns, perfectly balanced target (50/50).

**Workflow:**
1. **EDA & Feature Selection** — Correlation heatmaps; retained features with `|r| > 0.10` against `Diabetes` target.
2. **Feature Engineering** — Constructed three composite risk scores:
   - **`Cardio_Risk_Score`**: Sum of cardiovascular comorbidities (`HighBP + HighChol + HeartDiseaseorAttack + Stroke`).
   - **`BMI_x_Age`**: Interaction capturing metabolic risk accumulation across aging cohorts.
   - **`Health_Burden`**: `GenHlth × (PhysHlth + 1)` — self-rated health weighted by limitation days.
3. **Modeling** — Random Forest baseline → `RandomizedSearchCV` tuning → XGBoost with engineered features.
4. **Threshold Calibration** — Lowered from `0.50` → `0.45` to boost diabetic Recall from **0.80 → 0.84**.

**Final Model Performance (XGBoost @ threshold 0.45):**

| Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: |
| 75.00% | 0.71 | **0.84** | 0.77 |

**Top Features (XGBoost importance):**
1. `HighBP` — ~40.3% relative importance
2. `Cardio_Risk_Score` — ~21.8%
3. `GenHlth` — ~16.0%

---

### 02 — Hypertension / CVD

**Dataset:** Kaggle [`sulianova/cardiovascular-disease-dataset`](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (`cardio_train.csv`) — semicolon-delimited; cleaned to **68,539 rows**.

**Data Integrity Audit:**
- Dropped `id` column and 24 exact duplicate rows.
- Filtered physiologically impossible blood pressure readings (`ap_hi`: 70–250, `ap_lo`: 40–160) and height/weight outliers.
- Converted `age` from days to years.

**Feature Engineering:**
- **`bmi`**: Derived from `weight` / `(height/100)²`.
- **`pulse_pressure`**: `ap_hi − ap_lo`.
- **`map` (Mean Arterial Pressure)**: `ap_lo + pulse_pressure / 3`.
- Dropped low-signal columns: `gender`, `alco`, `height`, `smoke`, `active`, `weight`, `ap_lo`.

**Modeling:**
- Random Forest baseline (`n_estimators=100`, `max_depth=10`).
- XGBoost tuned via `RandomizedSearchCV` across depth, learning rate, regularization (L1/L2), subsampling, and `min_child_weight`.

**Final Model Performance (XGBoost @ threshold 0.42):**

| ROC-AUC | Recall | Accuracy |
| :---: | :---: | :---: |
| **0.802** | **0.76** | **73.5%** |

---

### 03 — Stroke

**Dataset:** Kaggle [`prosperchuks/health-dataset`](https://www.kaggle.com/datasets/prosperchuks/health-dataset) (`stroke_data.csv`) — severe class imbalance (~18:1 non-stroke to stroke).

**Data Integrity Audit:**
- Identified synthetic row-cloning: identical clinical profiles (`avg_glucose_level`, `bmi`, `hypertension`, etc.) were duplicated with randomized `age` and `sex` values — a form of target leakage.
- Dropped `age` and `sex` columns, then de-duplicated on remaining clinical features → **3,425 unique rows**.

**Feature Engineering:**
- **`Metabolic_Risk`**: `avg_glucose_level × bmi`
- **`Glucose_Heart_Risk`**: `avg_glucose_level × (heart_disease + 1)`
- **`Vascular_Risk`**: Composite of top 3 vascular predictors.
- Dropped low-signal lifestyle columns: `Residence_type`, `work_type`, `smoking_status`.

**Modeling:**
- `RandomForestClassifier(class_weight='balanced')` — chosen over XGBoost due to extreme imbalance.
- Stratified train/test split (`stratify=y`, `random_state=42`).

**Threshold Calibration:**
- Applied **Youden's J statistic** (`argmax(tpr - fpr)`) on the ROC curve → optimal threshold **`0.373`**.

**Final Model Performance (Random Forest @ threshold 0.373):**

| Recall (Stroke) | Note |
| :---: | :--- |
| **0.72** | Achieved on 18:1 imbalanced data after Youden's J calibration |

---

## Key Engineering Highlights

- **Synthetic Data & Leakage Auditing:** Identified and stripped synthetic row-cloning and randomized `age`/`sex` noise in community datasets, preventing target leakage and inflated validation scores.
- **Clinical Feature Engineering:** Constructed physiological interaction features including **BMI**, **Pulse Pressure**, **Mean Arterial Pressure (MAP)**, **Metabolic Risk**, and **Vascular Burden**.
- **Medical Threshold Tuning:** Shifted away from default `0.50` probability boundaries using ROC-AUC curves and **Youden's J statistic** to prioritize sensitivity (recall) and reduce missed diagnoses (False Negatives).
- **Class Imbalance Handling:** Used `class_weight='balanced'` in Random Forest and stratified splits for the stroke dataset to handle an extreme 18:1 minority class ratio.

---

## Project Structure

```text
AI-diabetes/
├── 01_diabetes_ml.ipynb      # Diabetes: EDA, feature engineering, XGBoost, threshold tuning
├── 02_hypertension_ml.ipynb  # Hypertension/CVD: audit, MAP/BMI engineering, XGBoost + RandomizedSearchCV
├── 03_stroke_ml.ipynb        # Stroke: leakage audit, balanced RF, Youden's J threshold
├── pyproject.toml            # Project metadata and dependencies (managed via uv)
├── uv.lock                   # Locked dependency tree
├── README.md                 # Project documentation
└── src/
    └── ai_diabetes/          # Source package directory
        └── __init__.py
```

---

## Installation & Setup

### Prerequisites
- Python `>= 3.12`
- Recommended: [`uv`](https://docs.astral.sh/uv/) (or standard `pip` / `venv`)
- Kaggle API credentials configured (datasets are downloaded automatically via `kagglehub`)

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

Launch Jupyter Lab to run any of the pipelines:

```bash
uv run jupyter lab
```

Open the desired notebook and run all cells. Each notebook will automatically download its dataset via `kagglehub`, perform data auditing, feature engineering, model training, and output performance metrics and visualizations.

| Notebook | Dataset downloaded |
| :--- | :--- |
| `01_diabetes_ml.ipynb` | `prosperchuks/health-dataset` → `diabetes_data.csv` |
| `02_hypertension_ml.ipynb` | `sulianova/cardiovascular-disease-dataset` → `cardio_train.csv` |
| `03_stroke_ml.ipynb` | `prosperchuks/health-dataset` → `stroke_data.csv` |

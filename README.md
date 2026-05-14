# Cluster-Based Student Profiling and Explainable Early Engagement Prediction Using OULAD

**Author:** Xiang Cui &nbsp;|&nbsp; **Supervisor:** Professor Karl Southern  
**Programme:** MSc Advanced Computer Science, Durham University  
**Final Submission:** 4 September 2026 &nbsp;|&nbsp; **Oral Exam:** Week of 14 September 2026

---

## Table of Contents

1. [Research Overview](#research-overview)
2. [Research Questions](#research-questions)
3. [Project Goals](#project-goals)
4. [Dataset](#dataset)
5. [Repository Structure](#repository-structure)
6. [Notebook Outline](#notebook-outline)
7. [Modelling Approach](#modelling-approach)
8. [Success Criteria](#success-criteria)
9. [Results Summary](#results-summary)
10. [Requirements](#requirements)
11. [Getting Started](#getting-started)

---

## Research Overview

The [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset) records 32,593 student module registrations across seven STEM courses delivered in 2013–2014, providing VLE clickstream logs, demographic attributes, and assessment records. Approximately 32% of students received a final outcome of *Withdrawn*, and a large proportion of these withdrawals occur during the early stages of a course — before any formal assessment signal can alert institutional staff.

This project addresses that early disengagement problem by:
1. Constructing interpretable student behavioural profiles via **k-Means clustering** on VLE interaction sequences.
2. Developing an **explainable XGBoost model** for early prediction of at-risk students.
3. Applying **per-cluster SHAP analysis** to reveal which behavioural drivers differ across learner archetypes.

**Target module:** FFF — selected over BBB for higher VLE coverage (92.2% vs 84.5%) and richer engagement signal (median 2,116 clicks vs 454).  
**Train / test split:** Temporal — 2013 presentations → train, 2014 presentations → test.

---

## Research Questions

**Central question:**  
> Can cluster-derived student behavioural profiles enable earlier and more interpretable at-risk prediction compared to raw-feature baselines?

**Sub-questions:**

| # | Sub-question |
|---|---|
| Q1 | Which behavioural features from OULAD VLE logs most effectively distinguish distinct student engagement profiles? |
| Q2 | Does profile-enriched input yield higher macro-F1 and AUC-ROC than the raw-feature XGBoost baseline? |
| Q3 | Do SHAP feature-importance rankings differ systematically across learner clusters, and which behavioural drivers are most predictive of at-risk outcomes within each profile? |

---

## Project Goals

| Tier | ID | Description | Risk |
|---|---|---|---|
| **Basic** | B1 | Train raw-feature XGBoost baseline on OULAD; report Accuracy, Precision, Recall, macro-F1, AUC-ROC | Low |
| **Basic** | B2 | Feature engineering from VLE logs; apply k-Means; validate with silhouette score; visualise clusters (PCA / t-SNE) and interpret profiles educationally | Low |
| **Intermediate** | I1 | Train profile-enriched XGBoost model (same architecture as B1, cluster label as additional feature); compare F1 and AUC-ROC to B1 | Medium |
| **Intermediate** | I2 | Apply SHAP to both models; produce comparative interpretability analysis | Medium |
| **Advanced** | A1 | Cross-module profile transferability study | High |
| **Advanced** | A2 | Multi-class engagement prediction | High |

---

## Dataset

OULAD consists of seven relational CSV tables:

| Table | Description |
|---|---|
| `studentInfo.csv` | Demographic and outcome data per student per module |
| `studentRegistration.csv` | Registration and unregistration dates |
| `studentAssessment.csv` | Per-student assessment scores and submission flags |
| `studentVle.csv` | Daily VLE click counts per resource |
| `assessments.csv` | Assessment metadata (type, weight, deadline) |
| `courses.csv` | Module and presentation duration |
| `vle.csv` | VLE resource metadata and activity types |

**Download:** https://analyse.kmi.open.ac.uk/open_dataset  
Place the raw CSV files in `../dataset/` relative to the notebook:

```
dataset/
├── assessments.csv
├── courses.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv
└── vle.csv
```

Cleaned outputs are written to `../dataset/cleaned/` automatically by the notebook.

---

## Repository Structure

```
oulad-at-risk-prediction/
├── OULAD.ipynb                                    # Main analysis and modelling notebook
├── requirements.txt                               # Python dependencies
├── README.md
└── Literature Review & Project Plan - Xiang Cui.pdf
```

---

## Notebook Outline

### Exploratory Data Analysis

| Section | Description |
|---|---|
| **1. Dataset Overview** | Table shapes, enrolment counts by module × presentation |
| **2. Data Cleaning** | Missing value audit across all 7 tables; documented imputation decisions |
| **3. Target Variable** | Final result distribution; withdrawal rate by module and presentation |
| **4. Learner Demographics** | Gender, age band, disability distribution vs. final outcome |
| **5. VLE Engagement** | Click and active-day distributions; per-module VLE coverage; no-VLE student analysis |
| **6. Assessment Behaviour** | Submission rates and score distributions by final result and assessment type |
| **7. Registration Timeline** | Registration and unregistration day distributions; early vs. late withdrawal |
| **8. Module Selection** | BBB vs. FFF comparison (enrolment, withdrawal rate, VLE coverage, click signal) |

### Goal B1 — Raw-Feature XGBoost Baseline

| Section | Description |
|---|---|
| **9. B1 — XGBoost Baseline** | Default XGBoost on raw features; temporal train/test evaluation |
| **10. imd_band Ablation** | Fairness-motivated removal of the socioeconomic deprivation index (`imd_band`) |
| **11. Default Model (no imd_band)** | Full evaluation of the baseline without `imd_band` |
| **12. RandomizedSearchCV Tuning** | 50-iteration random search × 5-fold stratified CV on 2013 train set |
| **13. Optuna Bayesian Optimisation** | 50-trial TPE search; 3-way comparison: Default vs. RandomizedSearchCV vs. Optuna |

### Goal B2 — k-Means Clustering & Learner Profiling *(planned)*

Feature engineering on VLE interaction sequences; k-Means with silhouette and Davies–Bouldin validation; PCA / t-SNE visualisation; educational interpretation of cluster profiles.

### Goals I1 & I2 — Profile-Enriched Model + SHAP *(planned)*

Cluster label injected as an additional feature into XGBoost; McNemar's test comparing I1 vs. B1; global and per-cluster SHAP beeswarm plots.

### Goals A1 & A2 — Advanced *(planned)*

Cross-module profile transferability study; multi-class engagement prediction.

---

## Modelling Approach

### Feature Engineering (B1)

| Feature Group | Features |
|---|---|
| VLE engagement | `total_clicks`, `active_days`, `mean_daily_clicks`, `max_daily_clicks`, `unique_resources` |
| Assessment | `mean_score`, `num_submissions` |
| Demographics | `gender`, `age_band`, `highest_education`, `disability` |

All values are aggregated over the full course presentation. Missing VLE/assessment values are filled with 0. `imd_band` is excluded following the fairness ablation in Section 10.

### Classification Model

**XGBoost** (`XGBClassifier`) with:
- `scale_pos_weight` = ratio of negative to positive class to handle ~30% withdrawal rate
- Temporal train/test split (2013 → train, 2014 → test) to prevent data leakage
- `macro-F1` as the primary optimisation metric

### Hyperparameter Search

| Strategy | Configurations | CV Folds | Total Fits |
|---|---|---|---|
| RandomizedSearchCV | 50 random | 5 stratified | 250 |
| Optuna TPE (Bayesian) | 50 trials | 5 stratified | 250 |

Search space: `n_estimators` [100–500], `max_depth` [3–8], `learning_rate` [0.01–0.30], `subsample` [0.60–1.00], `colsample_bytree` [0.60–1.00], `min_child_weight` [1–10], `gamma` [0–5], `reg_alpha` [0–1], `reg_lambda` [0.5–3.0].

---

## Success Criteria

### Q1 — Clustering Quality (Goal B2)

- **Silhouette Score > 0.4** — well-separated, cohesive clusters
- **Davies–Bouldin Index** — secondary internal metric (lower is better)
- **Educational interpretability** — each cluster maps to a recognisable learner archetype (e.g., high-engagement, intermittent, disengaged), validated via per-cluster feature distributions and PCA / t-SNE visualisation

### Q2 — Predictive Improvement (Goals B1 vs. I1)

- McNemar's test on paired prediction vectors to determine whether I1 yields a statistically significant improvement in macro-F1 and AUC-ROC over B1
- **Early-warning simulation:** training data restricted to the first N weeks (N ∈ {2, 4, 6}) to assess whether cluster profiles remain informative at earlier intervention windows

### Q3 — Interpretability (Goal I2)

- **Global SHAP rankings:** compare feature-importance rankings between B1 and I1 to quantify the explanatory value added by the cluster label
- **Per-cluster beeswarm plots:** identify which behavioural features drive at-risk predictions differently across learner profiles


## Requirements

- Python 3.X
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- optuna
- scipy
- ipywidgets (for Optuna progress bar)

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna scipy ipywidgets
```

---

## Getting Started

1. Download the OULAD dataset and place the CSV files in `../dataset/`.
2. Open `OULAD.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
3. Run all cells in order (Kernel → Restart & Run All).
4. Cleaned data will be saved to `../dataset/cleaned/`.

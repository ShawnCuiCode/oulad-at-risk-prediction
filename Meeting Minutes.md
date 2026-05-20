# Meeting Minutes

---

## Meeting 1 — 7 May 2026

**Project:** Cluster-Based Student Profiling and Explainable Early Engagement Prediction Using OULAD

**Topics covered:** Literature Review feedback; Exploratory Data Analysis progress; two methodology questions.

---

### Literature Review Revisions

Four revisions are required:

1. All abbreviations must be expanded on first use (e.g., VLE, OULAD, SHAP).
2. A dedicated section on datasets used in reviewed studies should be added.
3. Literature from education theory should be incorporated alongside technical papers.
4. Each reviewed paper must describe its research objective, dataset, and target variable — not accuracy figures alone.

Additional requirements:
- A summary comparison table covering all reviewed studies is required.
- Cluster archetype definitions must be supported by educational theory citations.

---

### Decisions

| Topic | Decision |
|---|---|
| `imd_band` | Feature may be included, but a comparative experiment (with vs. without) is required to quantify its contribution and support the ethical discussion. |
| Temporal validation | **Strategy A approved** — 2013B + 2013J as training set; 2014B + 2014J as test set. |

---

### Supervisor Recommendation

For the project demo, produce a **week-by-week performance curve** (weeks 2, 4, 6, 8, and 10) plotting AUC-ROC and macro-F1 against week number, to demonstrate at which point in the module the model becomes sufficiently reliable for intervention.

---

## Meeting 2 — 14 May 2026

**Project:** Cluster-Based Student Profiling and Explainable Early Engagement Prediction Using OULAD

**Topics covered:** Three pre-submitted methodology questions; general progress feedback.

---

### Decisions

| Topic | Decision |
|---|---|
| `imd_band` removal | **Confirmed — exclude from subsequent models.** Removal yields negligible performance change (macro-F1 Δ ≈ +0.0002, AUC Δ ≈ +0.002); data quality concerns apply. Rationale must be documented in the methodology section. |
| Label distribution shift | **No active correction required.** The withdrawal rate increase from 27.9% (2013 train) to 34.1% (2014 test) — a 6.2 pp gap — reflects the Open University's atypical educational model. With only two years of data it is not possible to identify which cohort is anomalous. This shift must be reported as a limitation of the temporal validation strategy. |
| B1 baseline configuration | **Retain both results.** Default-parameter (macro-F1 = 0.7892) and tuned (macro-F1 = 0.8183) results should be presented side by side when comparing against B2 and I1. Consolidation to a single baseline may be deferred to the results chapter based on presentation clarity. |

---

### Supervisor Recommendation

Prioritise the combination of **static student characteristics** with **engagement statistics derived at multiple time steps** following the registration date. Static features alone are considered insufficient; incorporating temporal engagement signals is expected to add meaningful predictive value.

---

## Meeting 3 — 20 May 2026

**Project:** Cluster-Based Student Profiling and Explainable Early Engagement Prediction Using OULAD

**Topics covered:** Completed clustering strand (B2); updated classification baseline (B1+); next implementation stage.

---

### Progress Reported

Three clustering models were built progressively:

| Model | Features | K selected |
|---|---|---|
| B2 | 7 | — |
| B2+ | 12 | 3 |
| B2-Ext | 16 | 3 |

---

### Decisions

| Topic | Decision |
|---|---|
| K = 3 | **Approved.** K = 2 achieves the best silhouette score but collapses distinct withdrawal trajectories, limiting interpretability. K = 4 yields the weakest DB score (1.50). K = 3 balances statistical validity and educational meaning; justification must be included in the report. |
| 16 features | **Appropriate.** The B2 → B2+ → B2-Ext progression is well-structured and the temporal features are well-motivated by the N-week prediction setting; rationale must be documented. |
| I1 ablation structure | **Feasible — proceed.** The four configurations (B1+ baseline, I1 main, I1-ablation, I1-K2) are confirmed. |
| SHAP | To be used for feature importance visualisation, presented alongside the I1 results. |

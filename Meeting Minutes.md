# Meeting Minutes

---

## Meeting 1 — 7 May 2026

This meeting covered progress on the dissertation, including feedback on the Literature Review and a report on Exploratory Data Analysis progress. Two specific methodology questions were also addressed, resulting in the following decisions and feedback.

The Literature Review requires four revisions:
(1) all abbreviations must be expanded on first use (e.g., VLE, OULAD, SHAP);
(2) a dedicated section on datasets used in reviewed studies should be added;
(3) literature from education theory should be incorporated alongside technical papers;
(4) each reviewed paper must describe its research objective, dataset, and target variable, rather than reporting accuracy figures alone.
A summary comparison table covering all reviewed studies is also required, and cluster archetype definitions must be supported by educational theory citations.

On `imd_band`, Professor Southern confirmed that the feature may be included, but a comparative experiment is required — training two model versions with and without `imd_band` — to quantify its contribution and support the ethical discussion.

On temporal validation, Strategy A was approved: Year 1 presentations (2013B + 2013J) as the training set and Year 2 presentations (2014B + 2014J) as the test set.

For the project demo, the supervisor recommended a week-by-week performance curve (weeks 2, 4, 6, 8, and 10) plotting AUC-ROC and F1 score against week number, to visually demonstrate at which point in the module the model becomes sufficiently reliable for intervention.

---

## Meeting 2 — 14 May 2026


This meeting addressed three pre-submitted methodology questions and included general progress feedback on the dissertation.

On `imd_band` removal, the exclusion of the feature was confirmed for all subsequent models. Removal yields negligible performance change (macro-F1 Δ ≈ +0.0002, AUC Δ ≈ +0.002) and data quality concerns apply. The rationale must be documented in the methodology section.

On label distribution shift, no active correction is required. The withdrawal rate increase from 27.9% (2013 train) to 34.1% (2014 test) — a 6.2 pp gap — reflects the Open University's atypical educational model. With only two years of data it is not possible to identify which cohort is anomalous. This shift must be reported as a limitation of the temporal validation strategy.

On the B1 baseline configuration, both results should be retained. The default-parameter result (macro-F1 = 0.7892) and the tuned result (macro-F1 = 0.8183) should be presented side by side when comparing against B2 and I1. Consolidation to a single baseline may be deferred to the results chapter based on presentation clarity.

The supervisor recommended prioritising the combination of static student characteristics with engagement statistics derived at multiple time steps following the registration date. Static features alone are considered insufficient; incorporating temporal engagement signals is expected to add meaningful predictive value.

---

## Meeting 3 — 20 May 2026

This meeting reviewed the completed clustering strand and the updated classification baseline, and confirmed the plan for the next implementation stage.

Three clustering models were built progressively in a structured B2 → B2+ → B2-Ext progression. B2 used 7 features as the initial model; B2+ extended this to 12 features with K = 3 selected; B2-Ext reached the final configuration of 16 temporal and static features with K = 3 confirmed.

On the choice of K = 3, this was approved. K = 2 achieves the best silhouette score but collapses distinct withdrawal trajectories, limiting interpretability. K = 4 yields the weakest DB score (1.50). K = 3 balances statistical validity and educational meaning, and the justification must be included in the report.

On the 16-feature set, the B2 → B2+ → B2-Ext progression was considered well-structured and the temporal features well-motivated by the N-week prediction setting; the rationale must be documented.

On the I1 ablation structure, the four configurations — B1+ baseline, I1 main, I1-ablation, and I1-K2 — were confirmed as feasible; implementation may proceed. SHAP will be used for feature importance visualisation and presented alongside the I1 results.

---

## Meeting 4 — 4 June 2026

The weekly progress update covering the I1 and I1+ integration models was submitted in advance along with two methodology questions.

The I1 and I1+ models were completed this week, injecting the K = 3 cluster label from B2+ into the XGBoost classifier. B1 → I1 yields Δmacro-F1 = +0.0078 and ΔAUC = +0.0069; B1+ → I1+ yields Δmacro-F1 = +0.0100 and ΔAUC = +0.0039. McNemar's test confirms I1+ vs B1 is statistically significant (p < 0.05). The supervisor noted that one model is better tuned than the others; if time permits, each model should be tuned independently and this should be mentioned in the report.

On framing the dissertation contribution, it is appropriate to position explainability as the primary contribution and performance improvement as a secondary finding. The cluster label provides a human-readable engagement profile attached to each prediction, allowing educators to understand why a student is flagged without inspecting individual feature values. Personal characteristics and engagement signals operate on two effectively different axes, and this distinction should be discussed in the results chapter.

---

## Meeting 5 — 11 June 2026

This meeting reviewed the SHAP analysis and natural language explanation work completed this week, and included general feedback on the dissertation direction and demo design.

The supervisor noted that personal characteristics such as gender and disability contribute very little to prediction, while engagement signals carry the main predictive weight. These operate on two effectively different axes, and this distinction should be discussed in the results chapter.

On the explanation output, more context and actual explanation is needed beyond numbers alone. Whoever uses the system may have no understanding of what the data means, so the output must be readable by a teacher without data science expertise. The explanation should convey why a student is at risk in natural language, not just present a risk label or raw feature values.

The supervisor recommended incorporating education-theory-based suggestions — not only explaining why a student is at risk, but also recommending what could be done to help them. This would extend the explainability argument and add practical value for educators.

On the demo, it does not need to show all work is finished. The teacher-facing interface should allow course selection, display a ranked list of at-risk students with their risk information, and provide individual student detail views. A1 can follow once the demo is in place.

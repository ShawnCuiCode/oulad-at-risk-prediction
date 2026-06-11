# -*- coding: utf-8 -*-
"""OULAD Student Withdrawal Risk -- Streamlit Demo (single page)
Run:  streamlit run code/app.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="OULAD Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "dataset")

# ── Helpers ─────────────────────────────────────────────────────
def risk_colour(p):
    if p >= 0.70: return "#e74c3c"
    if p >= 0.40: return "#f39c12"
    return "#27ae60"

def risk_emoji(p):
    if p >= 0.70: return "🔴 HIGH"
    if p >= 0.40: return "🟡 MEDIUM"
    return "🟢 LOW"

# ── Load models (once) ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    model     = joblib.load(os.path.join(SCRIPT_DIR, "model_i1.pkl"))
    scaler    = joblib.load(os.path.join(SCRIPT_DIR, "scaler.pkl"))
    km        = joblib.load(os.path.join(SCRIPT_DIR, "km_final.pkl"))
    explainer = joblib.load(os.path.join(SCRIPT_DIR, "explainer_i1.pkl"))
    with open(os.path.join(SCRIPT_DIR, "model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, km, explainer, meta

# ── Load raw data (once) ─────────────────────────────────────────
@st.cache_data(show_spinner="Loading OULAD dataset...")
def load_data():
    si  = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"))
    sv  = pd.read_csv(os.path.join(DATA_DIR, "studentVle.csv"))
    sa  = pd.read_csv(os.path.join(DATA_DIR, "studentAssessment.csv"))
    vle = pd.read_csv(os.path.join(DATA_DIR, "vle.csv"))
    sr  = pd.read_csv(os.path.join(DATA_DIR, "studentRegistration.csv"))
    return si, sv, sa, vle, sr

# ── Feature engineering (once, full dataset) ────────────────────
@st.cache_data(show_spinner="Engineering features...")
def build_all_features(_si, _sv, _sa, _vle, _sr, _scaler, _km, meta):
    fill_cols        = meta["fill_cols"]
    cat_cols         = meta["cat_cols"]
    feature_cols     = meta["feature_cols"]
    cluster_features = meta["cluster_features"]
    cat_enc          = meta["cat_encodings"]

    interactive_types = {"forumng","oucollaborate","ouwiki","ouelluminate","chat","questionnaire"}
    assessment_types  = {"quiz","externalquiz"}

    sv_merged = _sv.merge(_vle[["id_site","activity_type"]], on="id_site", how="left")

    def agg_vle(g):
        return pd.Series({
            "total_clicks"           : g["sum_click"].sum(),
            "active_days"            : g["date"].nunique(),
            "mean_daily_clicks"      : g["sum_click"].mean(),
            "max_daily_clicks"       : g["sum_click"].max(),
            "std_daily_clicks"       : g["sum_click"].std(ddof=0),
            "unique_resources"       : g["id_site"].nunique(),
            "interactive_clicks"     : g.loc[g["activity_type"].isin(interactive_types), "sum_click"].sum(),
            "assessment_clicks"      : g.loc[g["activity_type"].isin(assessment_types),  "sum_click"].sum(),
            "content_clicks"         : g.loc[~g["activity_type"].isin(interactive_types | assessment_types), "sum_click"].sum(),
            "activity_type_diversity": g["activity_type"].nunique(),
            "first_access_day"       : g["date"].min(),
            "last_access_day"        : g["date"].max(),
        })

    vle_feats = sv_merged.groupby("id_student").apply(agg_vle).reset_index()
    vle_feats["active_span"]      = vle_feats["last_access_day"] - vle_feats["first_access_day"]
    vle_feats["std_daily_clicks"] = vle_feats["std_daily_clicks"].fillna(0)

    assess_feats = _sa.groupby("id_student").agg(
        mean_score=("score","mean"), num_submissions=("id_assessment","count")
    ).reset_index()

    reg = _sr.groupby("id_student")["date_registration"].mean().reset_index()
    reg.columns = ["id_student","reg_offset"]

    df = (
        _si[["id_student","gender","age_band","highest_education","imd_band","disability","final_result"]]
        .drop_duplicates("id_student", keep="first")
        .merge(vle_feats,    on="id_student", how="left")
        .merge(assess_feats, on="id_student", how="left")
        .merge(reg,          on="id_student", how="left")
    )
    extra = ["interactive_clicks","assessment_clicks","content_clicks",
             "activity_type_diversity","reg_offset","active_span",
             "std_daily_clicks","first_access_day","last_access_day"]
    df[fill_cols] = df[fill_cols].fillna(0)
    for c in extra:
        if c in df.columns: df[c] = df[c].fillna(0)

    for c in cat_cols:
        enc_map = {v: int(k) for k, v in cat_enc[c].items()}
        df[c] = df[c].map(enc_map).fillna(-1).astype(int)

    clip_caps = meta.get("clip_caps", {})
    clust_df  = df.copy()
    for f, cap in clip_caps.items():
        if f in clust_df.columns: clust_df[f] = clust_df[f].clip(upper=cap)
    for cf in cluster_features:
        if cf not in clust_df.columns: clust_df[cf] = 0

    clust_vals   = clust_df[cluster_features].values
    clust_scaled = _scaler.transform(clust_vals)
    df["cluster"] = _km.predict(clust_scaled)

    i1_cols = meta.get("i1_feature_cols", feature_cols + ["cluster"])
    for c in i1_cols:
        if c not in df.columns: df[c] = 0
    if "cluster_p" in i1_cols:
        soft = _km.transform(clust_scaled)
        inv  = 1.0 / (soft + 1e-9)
        df["cluster_p"] = (inv / inv.sum(axis=1, keepdims=True))[:, 0]

    X      = df[i1_cols].values.astype(float)
    shap_c = feature_cols + ["cluster"]
    for c in shap_c:
        if c not in df.columns: df[c] = 0
    X_shap = df[shap_c].values.astype(float)

    return df, X, i1_cols, X_shap, shap_c

# ── Natural-language reason (educator-facing, no numeric values) ─
def generate_reason(sv_arr, feat_names, feat_vals, meta, prob=None):
    """
    Returns (diagnosis: str, recommendation: str).
    All output is plain English intended for teaching staff.
    No SHAP values, model scores, or raw counts are shown.
    """
    cat_decode = {c: {int(k): v for k, v in m.items()} for c, m in meta["cat_encodings"].items()}

    # ── Cluster plain-language descriptions ──────────────────────────────
    # Source: cluster behaviour profiles derived from OULAD 2013 cohort analysis
    _CP_DESC = {
        0: ("showing a moderate level of online engagement",
            "Students at this engagement level sometimes withdraw, "
            "particularly when other warning signs are also present."),
        1: ("showing very little engagement with online course materials",
            "Students who rarely use the online learning platform are among "
            "those most likely to withdraw before completing their course."),
        2: ("actively and consistently using the online learning platform",
            "Students with this pattern of engagement almost always "
            "complete their course successfully."),
        3: ("with no recorded activity on the online learning platform at all",
            "Students who never access course materials online almost "
            "always withdraw — this is the strongest single warning sign."),
    }

    # ── Plain-language feature descriptions (risk-raising) ───────────────
    def _risk_phrase(fname, fval):
        """Return a plain sentence describing why this feature raises concern, or None."""
        if fname == "num_submissions":
            if fval == 0:
                return "has not submitted any coursework to date"
            if fval < 3:
                return "has submitted very little coursework so far"
            if fval < 5:
                return "has completed fewer assignments than expected at this stage"
        elif fname in ("total_clicks", "active_days", "mean_daily_clicks", "interactive_clicks"):
            if fval == 0:
                return "has not logged into or used the online learning platform at all"
            if fname == "active_days" and fval < 5:
                return "has only logged into the platform on a handful of days"
            if fname in ("total_clicks", "mean_daily_clicks"):
                return "has made very limited use of the online learning platform"
        elif fname == "unique_resources":
            if fval == 0:
                return "has not opened any course materials or resources online"
            if fval < 5:
                return "has explored very few course resources online"
        elif fname == "mean_score":
            if fval == 0:
                return "has no recorded assessment results yet"
            if fval < 40:
                return "is struggling significantly with assessments"
            if fval < 60:
                return "is currently achieving below the expected standard in assessments"
        elif fname == "assessment_clicks":
            if fval == 0:
                return "has not accessed any practice quizzes or online assessments"
        elif fname == "content_clicks":
            if fval == 0:
                return "has not opened any study content on the platform"
        elif fname == "disability" and fval == 1:
            return "has a registered disability and may need additional support to participate fully"
        elif fname == "highest_education" and fval <= 1:
            return ("is entering the course without traditional prior qualifications, "
                    "which can make the academic workload more challenging")
        elif fname == "reg_offset" and fval > 30:
            return ("enrolled after the course had already started and may have missed "
                    "important early introductory sessions and peer connections")
        elif fname == "age_band":
            return ("is in an age group where students sometimes find it harder "
                    "to balance study with other life commitments")
        return None

    def _protect_phrase(fname, fval):
        """Return a plain sentence describing a protective factor, or None."""
        if fname == "num_submissions" and fval >= 5:
            return "has been keeping up well with coursework submissions"
        if fname == "mean_score" and fval >= 70:
            return "is performing well in assessments"
        if fname in ("total_clicks", "active_days") and fval >= 20:
            return "is actively and regularly using the online learning platform"
        if fname == "unique_resources" and fval >= 20:
            return "has been exploring a wide range of course materials"
        return None

    # ── Locate cluster feature ────────────────────────────────────────────
    cluster_fname = next((f for f in ("cluster", "cluster_p") if f in feat_names), None)
    cp_shap = cp_val = 0
    if cluster_fname:
        cp_fi   = feat_names.index(cluster_fname)
        cp_shap = float(sv_arr[cp_fi])
        cp_val  = int(round(float(feat_vals[cp_fi])))

    cp_engagement, cp_context = _CP_DESC.get(cp_val, (
        "with an unclear engagement pattern",
        "This student's online activity pattern does not clearly match any known group."
    ))

    # ── Rank non-cluster features by |SHAP|, collect phrases ─────────────
    order = sorted(range(len(feat_names)),
                   key=lambda i: abs(sv_arr[i]), reverse=True)
    risk_phrases    = []
    protect_phrases = []
    intervention_hints = []

    for fi in order:
        fname  = feat_names[fi]
        sv_val = float(sv_arr[fi])
        fval   = float(feat_vals[fi])
        if fname == cluster_fname:
            continue

        if sv_val > 0:
            phrase = _risk_phrase(fname, fval)
            if phrase and len(risk_phrases) < 3:
                risk_phrases.append(phrase)
                # Collect hints for recommendation
                if fname in ("total_clicks", "active_days", "mean_daily_clicks",
                             "unique_resources", "interactive_clicks", "content_clicks"):
                    intervention_hints.append("vle_absent" if fval == 0 else "vle_low")
                elif fname == "num_submissions":
                    intervention_hints.append("submit_missing" if fval == 0 else "submit_low")
                elif fname == "mean_score":
                    intervention_hints.append("score_zero" if fval == 0 else "score_low")
                elif fname == "disability" and fval == 1:
                    intervention_hints.append("disability")
                elif fname == "highest_education" and fval <= 1:
                    intervention_hints.append("education_low")
                elif fname == "reg_offset" and fval > 30:
                    intervention_hints.append("late_reg")
        else:
            phrase = _protect_phrase(fname, fval)
            if phrase and len(protect_phrases) < 1:
                protect_phrases.append(phrase)

    # ── Assemble diagnosis (no numbers, no technical terms) ──────────────
    # Para 1: engagement pattern + what it means historically
    diag_cluster = (
        f"This student's online learning behaviour is **{cp_engagement}**. "
        f"{cp_context}"
    )

    # Para 2: specific concerns and reassurances
    concern_parts = []
    if risk_phrases:
        if len(risk_phrases) == 1:
            concern_parts.append(f"In addition, the student {risk_phrases[0]}")
        elif len(risk_phrases) == 2:
            concern_parts.append(
                f"In addition, the student {risk_phrases[0]} and {risk_phrases[1]}")
        else:
            concern_parts.append(
                f"In addition, the student {', '.join(risk_phrases[:-1])}, "
                f"and {risk_phrases[-1]}")

    if protect_phrases:
        reassurance = f"On a positive note, the student {protect_phrases[0]}."
        concern_parts.append(reassurance)

    diag_details = "  \n".join(concern_parts) if concern_parts else ""
    diagnosis = "  \n".join(filter(None, [diag_cluster, diag_details]))

    # ── Build recommendation with educator-facing rationale ──────────────
    hint_set = set(intervention_hints)
    actions  = []
    bases    = []

    if "vle_absent" in hint_set:
        actions.append(
            "Contact this student directly as soon as possible — a personal phone call "
            "or email from their tutor is often enough to re-engage a student who has "
            "drifted away before they make a formal decision to withdraw.")
        bases.append(
            "Students who have never logged into the course platform almost never "
            "complete their module without a direct, personal outreach from teaching staff.")
    elif "vle_low" in hint_set:
        actions.append(
            "Reach out to the student and encourage them to set aside regular time "
            "each week to engage with course materials online. Offering a simple "
            "weekly study plan can significantly improve follow-through.")
        bases.append(
            "Regular use of online learning materials is one of the clearest indicators "
            "of whether a student will complete their course. Students who engage "
            "consistently are far less likely to withdraw.")

    if "submit_missing" in hint_set or "score_zero" in hint_set:
        actions.append(
            "Urgently remind the student about any outstanding assignments and offer "
            "a brief one-to-one meeting to help them understand what is expected. "
            "Even a partial submission shows commitment and can be a turning point.")
        bases.append(
            "Failure to submit any coursework is the single strongest indicator that "
            "a student is about to withdraw. In our analysis of thousands of students, "
            "those who never submitted an assignment almost always left the course.")
    elif "submit_low" in hint_set or "score_low" in hint_set:
        actions.append(
            "Encourage the student to attend a tutorial or drop-in session, or connect "
            "them with peer study groups. Students who feel they are falling behind "
            "academically are at high risk of giving up if they do not receive support.")
        bases.append(
            "Consistent assessment submission and reasonable performance are strong "
            "predictors of course completion. Students who begin to fall behind "
            "academically often withdraw unless they receive targeted academic support.")

    if cp_shap > 0 and cp_val in (1, 3):
        actions.append(
            "Consider assigning this student a named personal tutor or peer mentor "
            "who can check in with them regularly and help them feel connected "
            "to the learning community.")
        bases.append(
            "Students who show little or no online activity rarely complete their "
            "module without active personal support. A consistent point of contact "
            "with a tutor makes a measurable difference to retention.")

    if "disability" in hint_set:
        actions.append(
            "Ensure that the disability support team has been in contact with this "
            "student and that any agreed adjustments are actually in place. "
            "Sometimes students do not follow up on support they are entitled to.")
        bases.append(
            "Students with disabilities can face invisible day-to-day barriers. "
            "Proactive contact from teaching staff to check that support is working "
            "is more effective than waiting for the student to ask for help.")

    if "education_low" in hint_set:
        actions.append(
            "Offer the student access to academic skills workshops or bridging "
            "resources, and reassure them that finding the course difficult at "
            "first is normal and that support is available.")
        bases.append(
            "Students entering higher education without traditional qualifications "
            "often underestimate how much support is available to them. "
            "Targeted early support significantly improves their chances of "
            "completing the course.")

    if "late_reg" in hint_set:
        actions.append(
            "Put together a short catch-up guide for this student so they know "
            "exactly what they missed at the start of the course and who they can "
            "speak to for help filling the gaps.")
        bases.append(
            "Starting a course late means missing introductory material and the "
            "chance to form early study relationships with peers — both of which "
            "are important for staying motivated. A structured catch-up plan "
            "addresses both of these risks.")

    if not actions:
        if prob is not None and prob >= 0.70:
            actions.append(
                "Schedule a welfare check-in with this student's personal tutor "
                "within the next week to discuss how they are finding the course "
                "and whether any support would help.")
            bases.append(
                "The overall pattern of this student's engagement and background "
                "places them in a group where withdrawal is considerably more "
                "likely than average. Early, personal contact is the most "
                "effective intervention at this stage.")
        else:
            actions.append(
                "Keep a watchful eye on this student's engagement over the coming "
                "weeks. If their activity or assessment performance declines further, "
                "move to a more direct form of support.")
            bases.append(
                "While this student is not yet in the highest-risk category, "
                "the warning signs are present. Monitoring now means you can "
                "act quickly if the situation deteriorates.")

    recommendation = " ".join(actions)
    if bases:
        recommendation += f"\n\n**Why this matters:** {' '.join(bases)}"

    return diagnosis, recommendation

# ── Per-week risk prediction (cached per week value) ────────────
@st.cache_data(show_spinner=False)
def predict_week(week, _sv, _sa, _vle, _df_base, _model, _scaler, _km, meta_str):
    """Re-derive risk scores using only VLE/assessment data up to week*7 days.
    Cluster labels are re-computed from week-filtered VLE features so that
    cluster and VLE metrics are always consistent."""
    m      = json.loads(meta_str)
    cutoff = week * 7
    i_types = {"forumng","oucollaborate","ouwiki","ouelluminate","chat","questionnaire"}
    a_types = {"quiz","externalquiz"}

    sv_w = _sv[_sv["date"] <= cutoff]
    sa_w = _sa[_sa["date_submitted"] <= cutoff]

    if len(sv_w):
        sv_m = sv_w.merge(_vle[["id_site","activity_type"]], on="id_site", how="left")
        def _agg(g):
            return pd.Series({
                "total_clicks"           : g["sum_click"].sum(),
                "active_days"            : g["date"].nunique(),
                "mean_daily_clicks"      : g["sum_click"].mean(),
                "max_daily_clicks"       : g["sum_click"].max(),
                "std_daily_clicks"       : g["sum_click"].std(ddof=0),
                "unique_resources"       : g["id_site"].nunique(),
                "interactive_clicks"     : g.loc[g["activity_type"].isin(i_types), "sum_click"].sum(),
                "assessment_clicks"      : g.loc[g["activity_type"].isin(a_types),  "sum_click"].sum(),
                "content_clicks"         : g.loc[~g["activity_type"].isin(i_types | a_types), "sum_click"].sum(),
                "activity_type_diversity": g["activity_type"].nunique(),
                "first_access_day"       : g["date"].min(),
                "last_access_day"        : g["date"].max(),
            })
        vf = sv_m.groupby("id_student").apply(_agg).reset_index()
        vf["active_span"]      = vf["last_access_day"] - vf["first_access_day"]
        vf["std_daily_clicks"] = vf["std_daily_clicks"].fillna(0)
    else:
        vf = pd.DataFrame(columns=["id_student"])

    if len(sa_w):
        af = sa_w.groupby("id_student").agg(
            mean_score=("score","mean"),
            num_submissions=("id_assessment","count")
        ).reset_index()
    else:
        af = pd.DataFrame(columns=["id_student"])

    # Exclude pre-computed cluster/VLE columns — we will re-compute them below
    keep = [c for c in ["id_student","gender","age_band","highest_education","imd_band",
                         "disability","reg_offset",
                         "final_result","withdrawn"]
            if c in _df_base.columns]
    wdf = _df_base[keep].drop_duplicates("id_student").copy()

    if len(vf) > 0:
        wdf = wdf.merge(vf, on="id_student", how="left")
    if len(af) > 0:
        wdf = wdf.merge(af, on="id_student", how="left")

    # ── Re-compute cluster from week-filtered VLE features ──────────────
    cluster_features = m["cluster_features"]
    clip_caps = m.get("clip_caps", {})
    clust_w = wdf.copy()
    for f, cap in clip_caps.items():
        if f in clust_w.columns:
            clust_w[f] = clust_w[f].clip(upper=cap)
    for cf in cluster_features:
        if cf not in clust_w.columns:
            clust_w[cf] = 0.0
        else:
            clust_w[cf] = clust_w[cf].fillna(0.0)
    clust_vals   = clust_w[cluster_features].values.astype(float)
    clust_scaled = _scaler.transform(clust_vals)
    wdf["cluster"] = _km.predict(clust_scaled)
    cn_map = {int(k): v for k, v in m["cluster_names"].items()}
    wdf["cluster_name"] = wdf["cluster"].map(cn_map).fillna("Unknown")
    i1_feature_cols = m.get("i1_feature_cols", m["feature_cols"] + ["cluster"])
    if "cluster_p" in i1_feature_cols:
        soft = _km.transform(clust_scaled)
        inv  = 1.0 / (soft + 1e-9)
        wdf["cluster_p"] = (inv / inv.sum(axis=1, keepdims=True))[:, 0]
    # ────────────────────────────────────────────────────────────────────

    i1_cols = i1_feature_cols
    for c in i1_cols:
        if c not in wdf.columns:
            wdf[c] = 0.0
        else:
            wdf[c] = wdf[c].fillna(0.0)

    X_w = wdf[i1_cols].values.astype(float)
    wdf["risk_prob"] = _model.predict_proba(X_w)[:, 1]
    return wdf

# ════════════════════════════════════════════════════════════════
# LOAD EVERYTHING ONCE
# ════════════════════════════════════════════════════════════════
model, scaler, km, explainer, meta = load_models()
si, sv_raw, sa_raw, vle_raw, sr_raw = load_data()
df, X, i1_cols, X_shap, shap_cols   = build_all_features(
    si, sv_raw, sa_raw, vle_raw, sr_raw, scaler, km, meta)

cn = {int(k): v for k, v in meta["cluster_names"].items()}
df["cluster_name"] = df["cluster"].map(cn).fillna("Unknown")
df["withdrawn"]    = (df["final_result"] == "Withdrawn").astype(int)
df["risk_prob"]    = model.predict_proba(X)[:, 1]

CLUSTER_COLOURS = {"Disengaged":"#e74c3c","Moderate":"#f39c12",
                   "High Engaged":"#27ae60","No VLE Record":"#95a5a6"}

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.title("🎓 OULAD Student Withdrawal Early Warning System")
st.caption("MSc ACS -- Durham University | Xiang Cui | 2026")
st.markdown(
    "Combines **k-Means** behavioural clustering with **XGBoost** withdrawal prediction "
    "and **SHAP** explainability on the Open University Learning Analytics Dataset."
)

m = meta.get("i1_metrics", {})
b = meta.get("b1_baseline_metrics", {})
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Accuracy",   f"{m.get('accuracy',0):.3f}")
c2.metric("Precision",  f"{m.get('precision_macro',0):.3f}")
c3.metric("Recall",     f"{m.get('recall_macro',0):.3f}")
c4.metric("F1 (macro)", f"{m.get('f1_macro',0):.3f}",
          delta=f"{m.get('f1_macro',0)-b.get('f1_macro',0):+.4f} vs baseline")
c5.metric("AUC-ROC",    f"{m.get('auc_roc',0):.3f}",
          delta=f"{m.get('auc_roc',0)-b.get('auc_roc',0):+.4f} vs baseline")

# ── Dataset Overview ─────────────────────────────────────────────
with st.expander("📊 Dataset Overview", expanded=True):
    total_students   = si["id_student"].nunique()
    total_vle_events = int(sv_raw["sum_click"].sum())
    total_assessments = len(sa_raw)
    withdrawal_rate  = (si["final_result"] == "Withdrawn").mean()

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Students",      f"{total_students:,}")
    d2.metric("VLE Interactions",    f"{total_vle_events:,}")
    d3.metric("Assessment Records",  f"{total_assessments:,}")
    d4.metric("Overall Withdrawal Rate", f"{withdrawal_rate:.1%}")

    ov1, ov2, ov3 = st.columns(3)

    # Final result distribution
    with ov1:
        result_counts = si["final_result"].value_counts()
        colours_result = {"Pass":"#27ae60","Distinction":"#2980b9",
                          "Fail":"#e67e22","Withdrawn":"#e74c3c"}
        fig_r, ax_r = plt.subplots(figsize=(4, 3.2))
        bars_r = ax_r.bar(result_counts.index, result_counts.values,
                          color=[colours_result.get(c,"#95a5a6") for c in result_counts.index])
        ax_r.bar_label(bars_r, fmt="%d", padding=3, fontsize=8)
        ax_r.set_title("Final Result Distribution", fontsize=10)
        ax_r.set_ylabel("Students"); ax_r.tick_params(axis="x", labelsize=8)
        ax_r.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_r); plt.close()

    # Gender × Withdrawal
    with ov2:
        gw = si.groupby(["gender","final_result"]).size().unstack(fill_value=0)
        gw_pct = gw.div(gw.sum(axis=1), axis=0) * 100
        fig_g, ax_g = plt.subplots(figsize=(4, 3.2))
        bottom = np.zeros(len(gw_pct))
        g_colours = {"Pass":"#27ae60","Distinction":"#2980b9",
                     "Fail":"#e67e22","Withdrawn":"#e74c3c"}
        for col in gw_pct.columns:
            ax_g.bar(gw_pct.index, gw_pct[col], bottom=bottom,
                     color=g_colours.get(col,"#95a5a6"), label=col)
            bottom += gw_pct[col].values
        ax_g.set_title("Outcome by Gender (%)", fontsize=10)
        ax_g.set_ylabel("Percentage")
        ax_g.legend(fontsize=7, loc="upper right")
        ax_g.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_g); plt.close()

    # Age band × Withdrawal rate
    with ov3:
        age_wd = si.groupby("age_band").apply(
            lambda g: (g["final_result"] == "Withdrawn").mean() * 100
        ).sort_index()
        fig_a, ax_a = plt.subplots(figsize=(4, 3.2))
        ax_a.bar(age_wd.index, age_wd.values, color="#e74c3c", alpha=0.75)
        for i, v in enumerate(age_wd.values):
            ax_a.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
        ax_a.set_title("Withdrawal Rate by Age Band", fontsize=10)
        ax_a.set_ylabel("Withdrawal Rate (%)"); ax_a.tick_params(axis="x", labelsize=8)
        ax_a.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_a); plt.close()

st.divider()

# ════════════════════════════════════════════════════════════════
# TABS (no repeated data loading)
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["⚠️ Early Warning", "🔍 Student Lookup", "📌 Cluster Profiles"])

# ── TAB 1: EARLY WARNING ─────────────────────────────────────────
with tab1:
    st.subheader("Top At-Risk Students by Teaching Week")
    col_w, col_n, _ = st.columns([1,1,3])
    week  = col_w.slider("Teaching week", 1, 26, 4, key="week_slider")
    top_n = col_n.selectbox("Show top N", [25, 50, 100], index=1)

    week_df = predict_week(week, sv_raw, sa_raw, vle_raw, df, model, scaler, km,
                           json.dumps(meta, sort_keys=True))

    proba_all = week_df["risk_prob"].values
    high   = (proba_all >= 0.70).sum()
    medium = ((proba_all >= 0.40) & (proba_all < 0.70)).sum()
    low    = (proba_all < 0.40).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 High Risk (>=70%)",    f"{high:,}")
    m2.metric("🟡 Medium Risk (40-70%)", f"{medium:,}")
    m3.metric("🟢 Low Risk (<40%)",      f"{low:,}")

    top_df = week_df.nlargest(top_n, "risk_prob").reset_index(drop=True)
    top_df.index += 1

    disp_cols = ["id_student","risk_prob","cluster_name",
                 "total_clicks","active_days","num_submissions","mean_score","final_result"]
    disp = top_df[[c for c in disp_cols if c in top_df.columns]].copy()
    disp.insert(2, "Risk", top_df["risk_prob"].apply(risk_emoji))
    disp = disp.rename(columns={
        "id_student":"Student ID","risk_prob":"Prob",
        "cluster_name":"Cluster","total_clicks":"VLE Clicks",
        "active_days":"Active Days","num_submissions":"Subs",
        "mean_score":"Avg Score","final_result":"Actual"})
    disp["Prob"] = disp["Prob"].map("{:.3f}".format)

    def row_colour(row):
        p = float(row["Prob"])
        if p >= 0.70: return ["background-color:#fde8e8"]*len(row)
        if p >= 0.40: return ["background-color:#fef9e7"]*len(row)
        return [""]*len(row)

    st.dataframe(disp.style.apply(row_colour, axis=1), height=480)

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.pie([high, medium, low],
               labels=["High (>=70%)", "Medium (40-70%)", "Low (<40%)"],
               colors=["#e74c3c","#f39c12","#27ae60"],
               autopct="%1.1f%%", startangle=90)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_bar:
        # Use model's default decision boundary (predict_proba >= 0.5)
        pred_labels   = (week_df["risk_prob"] >= 0.5).astype(int)
        actual_labels = (week_df["final_result"] == "Withdrawn").astype(int)
        wd_total   = int(actual_labels.sum())
        wd_correct = int(((pred_labels == 1) & (actual_labels == 1)).sum())
        flagged    = int(pred_labels.sum())
        accuracy   = float((pred_labels == actual_labels).mean())
        recall     = wd_correct / wd_total if wd_total > 0 else 0.0
        precision  = wd_correct / flagged  if flagged  > 0 else 0.0

        t1, t2, t3 = st.columns(3)
        t1.metric("Predicted At-Risk",  f"{flagged:,}",
                  help="Students with risk_prob ≥ 0.5 (model default)")
        t2.metric("Withdrawals Caught", f"{wd_correct:,} / {wd_total:,}",
                  delta=f"{recall:.1%} recall")
        t3.metric("Week Accuracy",      f"{accuracy:.1%}")

        # Prob distribution histogram for this week
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.hist(week_df["risk_prob"], bins=30, color="#3498db", edgecolor="white", alpha=0.8)
        ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Model boundary (0.5)")
        ax2.set_xlabel("Predicted withdrawal probability")
        ax2.set_ylabel("Number of students")
        ax2.set_title(f"Risk Score Distribution  (Week {week})")
        ax2.legend(fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        if week <= 3:
            st.caption(
                f"⚠️ Week {week} accuracy is limited: only {week * 7} days of VLE/assessment data "
                "are available. The model gains much more signal after week 4 when assessments begin."
            )
        else:
            st.caption("**Tip:** Enter a Student ID in the Student Lookup tab for a full SHAP explanation.")

# ── TAB 2: STUDENT LOOKUP ────────────────────────────────────────
with tab2:
    st.subheader("Individual Student Risk Profile")

    sample_ids = si["id_student"].sample(5, random_state=42).tolist()
    st.caption(f"Sample IDs to try: {', '.join(str(x) for x in sample_ids)}")

    student_id = st.number_input("Student ID", min_value=1, value=sample_ids[0], step=1)

    if st.button("Look Up", type="primary"):
        mask = df["id_student"] == student_id
        if not mask.any():
            st.error(f"Student ID {student_id} not found.")
        else:
            idx          = df[mask].index[0]
            prob         = float(df.loc[idx, "risk_prob"])
            cluster_name = df.loc[idx, "cluster_name"]
            actual_str   = " / ".join(si[si["id_student"]==student_id]["final_result"].unique())

            a1, a2, a3 = st.columns(3)
            a1.metric("Withdrawal Probability", f"{prob:.3f}")
            a2.metric("Behavioural Cluster",    cluster_name)
            a3.metric("Actual Outcome",         actual_str)

            colour = risk_colour(prob)
            st.markdown(
                f'<div style="background:#eee;border-radius:8px;height:26px;width:100%">'
                f'<div style="background:{colour};border-radius:8px;height:26px;'
                f'width:{prob*100:.1f}%;text-align:center;color:white;font-weight:bold;line-height:26px">'
                f'{risk_emoji(prob)} -- {prob*100:.1f}%</div></div>',
                unsafe_allow_html=True
            )
            st.markdown("")

            snap_keys  = ["total_clicks","active_days","mean_daily_clicks",
                          "unique_resources","mean_score","num_submissions"]
            snap_label = ["VLE Clicks","Active Days","Clicks/Day",
                          "Resources","Avg Score","Submissions"]
            sc = st.columns(len(snap_keys))
            for col, key, label in zip(sc, snap_keys, snap_label):
                val = df.loc[idx, key] if key in df.columns else 0
                col.metric(label, f"{val:.1f}")

            st.subheader("SHAP Feature Explanation")
            xi_shap = X_shap[idx:idx+1]
            sv_arr  = explainer.shap_values(xi_shap)[0]
            base    = float(explainer.expected_value)
            pairs   = sorted(zip(shap_cols, sv_arr, X_shap[idx]),
                             key=lambda x: abs(x[1]), reverse=True)[:12]
            names   = [p[0] for p in pairs]
            values  = [p[1] for p in pairs]
            bcolors = ["#e74c3c" if v > 0 else "#27ae60" for v in values]

            fig, ax = plt.subplots(figsize=(8,5))
            ax.barh(names[::-1], values[::-1], color=bcolors[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value  (positive = raises risk, negative = lowers risk)")
            ax.set_title(f"SHAP Explanation -- Student {student_id}")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            diagnosis, recommendation = generate_reason(
                sv_arr, shap_cols, X_shap[idx], meta, prob=prob)
            st.info(f"**Key Insight:** {diagnosis}")
            st.warning(f"**Recommended Action:** {recommendation}")

            shap_tbl = pd.DataFrame({
                "Feature"      : shap_cols,
                "SHAP Value"   : [f"{v:+.4f}" for v in sv_arr],
                "Feature Value": [f"{v:.2f}"  for v in X_shap[idx]],
            }).sort_values("SHAP Value", key=lambda s: s.str.replace("+","").astype(float).abs(),
                           ascending=False)
            st.dataframe(shap_tbl.reset_index(drop=True))
            st.caption(f"Baseline prob: {base:.3f}  -->  Predicted: {prob:.3f}")

# ── TAB 3: CLUSTER PROFILES ──────────────────────────────────────
with tab3:
    st.subheader("Learner Cluster Profiles")

    cluster_counts = df["cluster_name"].value_counts()
    wd_by_cluster  = df.groupby("cluster_name")["withdrawn"].mean().sort_values(ascending=False)
    ordered        = wd_by_cluster.index.tolist()

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5,3.5))
        bars = ax.bar(cluster_counts.index, cluster_counts.values,
                      color=[CLUSTER_COLOURS.get(c,"#3498db") for c in cluster_counts.index])
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_title("Cluster Sizes"); ax.set_ylabel("Students")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(5,3.5))
        ax2.bar(wd_by_cluster.index, wd_by_cluster.values*100,
                color=[CLUSTER_COLOURS.get(c,"#3498db") for c in wd_by_cluster.index])
        ax2.set_title("Withdrawal Rate by Cluster"); ax2.set_ylabel("Withdrawal Rate (%)")
        ax2.set_ylim(0, 100)
        for i, v in enumerate(wd_by_cluster.values):
            ax2.text(i, v*100+1.5, f"{v:.1%}", ha="center", fontsize=10)
        ax2.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    eng_cols = [c for c in ["total_clicks","active_days","mean_score",
                             "num_submissions","unique_resources","mean_daily_clicks"]
                if c in df.columns]
    cluster_means = df.groupby("cluster_name")[eng_cols].mean()
    st.dataframe(
        cluster_means.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.1f}"),
        use_container_width=True
    )

    plot_df = df[df["total_clicks"] < df["total_clicks"].quantile(0.95)]
    groups  = [plot_df[plot_df["cluster_name"]==c]["total_clicks"].dropna() for c in ordered]
    fig3, ax3 = plt.subplots(figsize=(7,4))
    bp = ax3.boxplot(groups, labels=ordered, patch_artist=True)
    for patch, label in zip(bp["boxes"], ordered):
        patch.set_facecolor(CLUSTER_COLOURS.get(label, "#3498db"))
        patch.set_alpha(0.7)
    ax3.set_title("VLE Clicks Distribution by Cluster")
    ax3.set_ylabel("Total VLE Clicks (95th pct capped)")
    ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

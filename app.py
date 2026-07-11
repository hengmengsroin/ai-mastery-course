"""
Module 1 — Interactive ML Pipeline Builder for Students
Applied AI Mastery Program | Rina Buoy, PhD

Upload any CSV → encode columns → split → tune → evaluate → export

Run:  streamlit run app.py
"""
import warnings; warnings.filterwarnings("ignore")
import io, pickle, textwrap

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except ImportError:
    UCI_AVAILABLE = False

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_digits, load_iris
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from scipy import stats as scipy_stats

from torch_mlp import TorchMLPClassifier, TorchMLPRegressor, TORCH_AVAILABLE

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Pipeline Builder", page_icon="🔬", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.35rem; }
div[data-testid="stExpander"] > div { padding: 4px 8px; }
</style>
""", unsafe_allow_html=True)

STEPS = ["Upload Data", "Configure Columns", "Split Data",
         "Train & Tune", "Evaluate", "Export"]

# Curated popular datasets from the UCI Machine Learning Repository (archive.ics.uci.edu)
# name -> UCI dataset id (see archive.ics.uci.edu/dataset/<id>)
UCI_POPULAR = {
    "Iris — classification":                         53,
    "Wine — classification":                         109,
    "Wine Quality — regression / classification":    186,
    "Breast Cancer Wisconsin (Diagnostic) — classification": 17,
    "Adult / Census Income — classification":        2,
    "Heart Disease — classification":                45,
    "Bank Marketing — classification":               222,
    "Car Evaluation — classification":               19,
    "Abalone — regression":                          1,
    "Auto MPG — regression":                          9,
    "Student Performance — regression":              320,
    "Concrete Compressive Strength — regression":    165,
    "Forest Fires — regression":                     162,
    "Ionosphere — classification":                   52,
    "Mushroom — classification":                     73,
    "Seeds — classification":                        236,
    "Spambase — classification":                     94,
}

# Small built-in text dataset (SMS-style spam/ham) — demonstrates the "Text (TF-IDF)"
# encoding option without needing a network download.
SAMPLE_TEXT_DATA = [
    ("Congratulations! You've won a $1000 gift card, claim now by calling us today", 1),
    ("URGENT: your account will be suspended unless you verify your details immediately", 1),
    ("You have been selected for a free cruise! Reply YES to claim your prize", 1),
    ("WINNER!! As a valued customer you receive a cash reward, call now to claim", 1),
    ("Get cheap meds online now, limited time offer, click the link immediately", 1),
    ("Your loan of $5000 has been approved! Confirm your bank details to receive funds", 1),
    ("FREE entry into our weekly draw, text WIN to enter, prizes worth up to $500", 1),
    ("Claim your free vacation package today, no strings attached, click here", 1),
    ("You've been chosen to receive a brand new iPhone, just pay shipping", 1),
    ("Act now! Your credit card has been charged, click to dispute this transaction", 1),
    ("Hot singles in your area want to meet you tonight, click to chat now", 1),
    ("Final notice: your subscription payment failed, update your card immediately", 1),
    ("Limited time offer, buy one get one free on all watches, shop now", 1),
    ("Your parcel could not be delivered, pay a small fee to reschedule delivery", 1),
    ("Congratulations, your number has won the lottery, contact our agent to claim", 1),
    ("Earn $5000 a week working from home, no experience needed, sign up today", 1),
    ("Your bank account has been locked, verify your identity now to restore access", 1),
    ("Exclusive deal just for you, huge discount on electronics, offer ends tonight", 1),
    ("You qualify for a government grant, apply now before the deadline expires", 1),
    ("Double your investment in 24 hours, guaranteed returns, start trading now", 1),
    ("Can we reschedule our meeting to 3pm tomorrow afternoon instead", 0),
    ("Thanks for dinner last night, I had a really great time with you", 0),
    ("Don't forget to pick up milk and eggs on your way home from work", 0),
    ("The quarterly report is attached, let me know if you have any questions", 0),
    ("Happy birthday! Hope you have a wonderful day surrounded by friends and family", 0),
    ("Are we still on for coffee this weekend, let me know what time works", 0),
    ("I finished the project draft, could you review it before Friday please", 0),
    ("The kids have soccer practice at five, can you pick them up today", 0),
    ("Just landed, will be home in about an hour, traffic looks light so far", 0),
    ("Reminder that rent is due at the end of the month as usual", 0),
    ("Great job on the presentation today, the client seemed really impressed", 0),
    ("Can you send me the notes from today's lecture when you get a chance", 0),
    ("Let's grab lunch tomorrow, there's a new place near the office", 0),
    ("The doctor's appointment got moved to next Tuesday at nine in the morning", 0),
    ("Thanks for helping me move last weekend, I really appreciate your time", 0),
    ("I'll be working from home tomorrow, call me if anything urgent comes up", 0),
    ("The flight got delayed by two hours, I'll text you when I land", 0),
    ("Loved the book you recommended, already starting the sequel this week", 0),
    ("Please remember to submit your timesheet before end of day Friday", 0),
    ("The weather looks great this weekend, want to go hiking on Saturday", 0),
]

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in dict(
    step=0, df=None, target_col=None, feature_cols=[], col_config_df=None,
    task="regression", preprocessor=None, feat_names_out=[],
    X_tr=None, X_te=None, y_tr=None, y_te=None,
    X_tr_raw=None, X_te_raw=None,
    model=None, best_params={}, cv_results_df=None, y_pred=None, y_prob=None,
    show_overview=True, uci_metadata=None, uci_variables=None, image_shape=None,
).items():
    if k not in st.session_state:
        st.session_state[k] = v

S = st.session_state

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ML Pipeline Builder")
    st.caption("Applied AI Mastery Program · Rina Buoy, PhD")
    if not S.show_overview:
        st.divider()
        for i, name in enumerate(STEPS):
            icon = "✅" if i < S.step else ("🔵" if i == S.step else "⬜")
            weight = "**" if i == S.step else ""
            st.markdown(f"{icon} &nbsp; {weight}{name}{weight}")
        st.divider()
        if S.step > 0 and st.button("← Back", use_container_width=True):
            S.step -= 1
            st.rerun()
    else:
        st.divider()
        st.markdown("Welcome to the interactive ML pipeline. Click **Start** to begin.")

def advance():
    S.step += 1

# ── Utilities ──────────────────────────────────────────────────────────────────

def is_numeric(series): return pd.api.types.is_numeric_dtype(series)

def detect_task(series):
    n = series.nunique()
    if is_numeric(series) and n > 15:
        return "regression"
    return "binary" if n == 2 else "multiclass"

def is_texty(series, min_avg_words=3, min_unique_ratio=0.5):
    """Heuristic: free-text column (long, mostly-unique strings) vs. a categorical label."""
    if is_numeric(series):
        return False
    s = series.dropna().astype(str)
    if s.empty:
        return False
    avg_words = s.str.split().str.len().mean()
    uniq_ratio = s.nunique() / len(s)
    return avg_words >= min_avg_words and uniq_ratio > min_unique_ratio

def detected_dtype(col, df):
    if is_numeric(df[col]):    return "numeric"
    if is_texty(df[col]):      return "text"
    return "categorical"

def default_encoding(col, df):
    if is_numeric(df[col]):  return "StandardScaler"
    if is_texty(df[col]):    return "Text (TF-IDF)"
    return "One-Hot"

def metric_row(d: dict):
    cols = st.columns(len(d))
    for col, (label, val) in zip(cols, d.items()):
        col.metric(label, f"{val:.4f}" if isinstance(val, float) else str(val))

def build_preprocessor(df, feature_cols, config_df):
    """Build ColumnTransformer from the config DataFrame (one row per feature col)."""
    cfg = config_df.set_index("Column")
    std, mm, keep, ohe = [], [], [], []
    for col in feature_cols:
        enc = cfg.loc[col, "Encoding"] if col in cfg.index else default_encoding(col, df)
        if enc == "StandardScaler":   std.append(col)
        elif enc == "MinMaxScaler":   mm.append(col)
        elif enc == "None (keep)":    keep.append(col)
        elif enc == "One-Hot":        ohe.append(col)

    transformers = []
    if std:  transformers.append(("std",  StandardScaler(), std))
    if mm:   transformers.append(("mm",   MinMaxScaler(),   mm))
    if keep: transformers.append(("pass", "passthrough",    keep))
    if ohe:  transformers.append(("ohe",
        OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ohe))

    for col in feature_cols:
        enc = cfg.loc[col, "Encoding"] if col in cfg.index else ""
        if enc == "Ordinal":
            order_str = str(cfg.loc[col, "Ordinal order (low → high)"] or "")
            order = [x.strip() for x in order_str.split(",") if x.strip()]
            if not order:
                order = sorted(df[col].dropna().astype(str).unique().tolist())
            transformers.append((f"ord_{col}",
                OrdinalEncoder(categories=[order],
                               handle_unknown="use_encoded_value", unknown_value=-1),
                [col]))
        elif enc == "Text (TF-IDF)":
            # Pass the column name (not [col]) so ColumnTransformer selects it as a
            # 1-D Series — TfidfVectorizer requires an iterable of raw strings.
            transformers.append((f"tfidf_{col}",
                TfidfVectorizer(max_features=200, stop_words="english",
                                 token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"),
                col))

    # sparse_threshold=0 forces a dense output matrix — TF-IDF is sparse by default,
    # but the rest of the app (previews, describe(), sample tables) assumes a dense array.
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

def mlp_diagram(layer_sizes, max_nodes_per_layer=12):
    """Draw a node-and-edge diagram for an MLP given [n_in, h1, h2, ..., n_out]."""
    display_sizes = [min(n, max_nodes_per_layer) for n in layer_sizes]
    node_pos = []
    for n_disp in display_sizes:
        ys = np.linspace(-(n_disp - 1) / 2, (n_disp - 1) / 2, n_disp) if n_disp > 1 else np.array([0.0])
        node_pos.append(ys)

    edge_x, edge_y = [], []
    for li in range(len(node_pos) - 1):
        for y0 in node_pos[li]:
            for y1 in node_pos[li + 1]:
                edge_x += [li, li + 1, None]
                edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(color="rgba(150,150,150,0.3)", width=0.6),
                              hoverinfo="skip", showlegend=False))

    n_layers = len(layer_sizes)
    colors = ["#1f77b4"] + ["#ff7f0e"] * (n_layers - 2) + ["#2ca02c"]
    labels = (["Input"] + [f"Hidden {i+1}" for i in range(n_layers - 2)] + ["Output"]
              if n_layers > 1 else ["Layer"])
    for li, ys in enumerate(node_pos):
        fig.add_trace(go.Scatter(
            x=[li] * len(ys), y=ys, mode="markers",
            marker=dict(size=16, color=colors[li], line=dict(color="white", width=1)),
            hovertext=[f"{labels[li]} node"] * len(ys), hoverinfo="text", showlegend=False))
        cap_note = f" (showing {display_sizes[li]})" if layer_sizes[li] > display_sizes[li] else ""
        fig.add_annotation(x=li, y=(ys.min() if len(ys) else 0) - 1.4,
                            text=f"<b>{labels[li]}</b><br>{layer_sizes[li]} units{cap_note}",
                            showarrow=False, font=dict(size=11), align="center")

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.6, n_layers - 0.4]),
        yaxis=dict(visible=False),
        height=380, margin=dict(t=30, b=70, l=20, r=20),
        title="Network Architecture", showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW — Methodology
# ══════════════════════════════════════════════════════════════════════════════
if S.show_overview:
    st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem 0">
  <h1 style="font-size:2.4rem;margin-bottom:.3rem">ML Pipeline Builder</h1>
  <p style="font-size:1.1rem;color:#666">
    Applied AI Mastery Program &nbsp;·&nbsp; Rina Buoy, PhD
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
This interactive app walks you through a complete **supervised machine learning workflow**
using scikit-learn. Upload any tabular dataset and follow the six steps below — or use one
of the ready-made sample CSV files to get started immediately.
""")

    st.divider()
    st.subheader("The 6-Step ML Methodology")

    steps_info = [
        ("1", "Upload Data",
         "Load any CSV file or a built-in dataset. Inspect column types, sample values, "
         "and missing data before modelling."),
        ("2", "Configure Columns",
         "Select the target (what to predict) and choose an encoding strategy for each "
         "feature: StandardScaler, MinMaxScaler, One-Hot, Ordinal, or drop."),
        ("3", "Split Data",
         "Divide the dataset into train / test sets with optional stratification. "
         "Inspect the encoded feature matrix and verify class balance."),
        ("4", "Train & Tune",
         "Pick a linear model (Linear/Ridge/Lasso/ElasticNet or Logistic Regression) "
         "and run GridSearchCV to find the best regularisation strength."),
        ("5", "Evaluate",
         "Review metrics (R², RMSE, AUC-ROC, F1 …), confusion matrix, ROC curve, "
         "coefficient chart, and a sample predictions table."),
        ("6", "Export",
         "Download the fitted model bundle (model.pkl) and predictions CSV, "
         "plus a ready-to-run Python deployment snippet."),
    ]

    row1 = st.columns(3)
    row2 = st.columns(3)
    for col, (num, title, desc) in zip(row1 + row2, steps_info):
        with col:
            st.markdown(f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:1rem;
            background:#fafafa;height:100%">
  <div style="font-size:1.6rem;font-weight:700;color:#1f77b4">Step {num}</div>
  <div style="font-weight:600;font-size:1rem;margin:.25rem 0 .5rem 0">{title}</div>
  <div style="font-size:.88rem;color:#555;line-height:1.5">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Key Concepts")
        st.markdown("""
| Concept | What you'll see |
|---|---|
| Feature engineering | Encoding, scaling, ordinal order |
| Bias-variance tradeoff | Train vs validation curves in CV |
| Regularisation | Ridge (L2), Lasso (L1), ElasticNet |
| Cross-validation | K-Fold / Stratified K-Fold GridSearchCV |
| Model evaluation | R², RMSE, AUC-ROC, confusion matrix |
| Decision threshold | Live slider → precision / recall tradeoff |
| Feature importance | Learned coefficient bar charts |
| Deployment | Pickle export + Python code snippet |
""")

    with col_r:
        st.subheader("Sample Datasets")
        st.markdown("""
| File | Task | Target | Rows |
|---|---|---|---|
| `housing.csv` | Regression | `price` | 800 |
| `churn.csv` | Binary classification | `churned` | 800 |
| `student_grades.csv` | Multiclass | `grade` (A/B/C/D) | 600 |
| `diabetes.csv` | Binary classification | `diabetes` | 750 |

Download the samples folder from the course materials, then upload any file in **Step 1**.
""")

    st.divider()
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("Start Building →", type="primary", use_container_width=True):
            S.show_overview = False
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Upload Data
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 0:
    st.title("Step 1 — Upload Your Dataset")

    c_up, c_sample, c_uci = st.columns([1, 1, 1.2])
    with c_up:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded:
            try:
                S.df = pd.read_csv(uploaded)
                S.uci_metadata = None; S.uci_variables = None; S.image_shape = None
                st.success(f"✓ {S.df.shape[0]:,} rows × {S.df.shape[1]} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    with c_sample:
        st.markdown("**— or use a built-in dataset —**")
        choice = st.selectbox("Sample dataset",
            ["(none)", "California Housing  (regression)",
             "Breast Cancer  (binary classification)",
             "Iris  (multiclass classification)",
             "Digits  (8×8 images, MNIST-like, multiclass)",
             "SMS Spam/Ham  (text, binary classification)"])
        if choice != "(none)" and st.button("Load sample"):
            S.image_shape = None
            if "California" in choice:
                ds = fetch_california_housing(as_frame=True); S.df = ds.frame.copy()
            elif "Breast" in choice:
                ds = load_breast_cancer(as_frame=True)
                S.df = ds.frame.copy(); S.df["target"] = 1 - S.df["target"]
            elif "Digits" in choice:
                ds = load_digits(as_frame=True)
                S.df = ds.frame.copy()
                S.image_shape = (8, 8)
            elif "SMS" in choice:
                S.df = pd.DataFrame(SAMPLE_TEXT_DATA, columns=["message", "label"])
            else:
                ds = load_iris(as_frame=True)
                S.df = ds.data.join(ds.target)
            S.uci_metadata = None; S.uci_variables = None
            st.success(f"✓ {S.df.shape[0]:,} rows × {S.df.shape[1]} columns")

    with c_uci:
        st.markdown("**— or connect a UCI ML Repository dataset —**")
        st.caption("Live from [archive.ics.uci.edu](https://archive.ics.uci.edu/)")
        if not UCI_AVAILABLE:
            st.warning("Install the `ucimlrepo` package to enable this: "
                       "`pip install ucimlrepo`")
        else:
            uci_choice = st.selectbox("Popular UCI datasets",
                ["(none)"] + list(UCI_POPULAR) + ["Custom dataset ID…"])
            uci_id = None
            if uci_choice == "Custom dataset ID…":
                uci_id = st.number_input(
                    "UCI dataset ID", min_value=1, value=53, step=1,
                    help="Found in the dataset's URL, e.g. "
                         "archive.ics.uci.edu/dataset/53/iris → ID is 53")
            elif uci_choice != "(none)":
                uci_id = UCI_POPULAR[uci_choice]

            if uci_id and st.button("Fetch from UCI"):
                try:
                    with st.spinner(f"Fetching dataset #{uci_id} from UCI…"):
                        ds = fetch_ucirepo(id=int(uci_id))
                        X, y = ds.data.features, ds.data.targets
                        S.df = X.join(y) if y is not None else X.copy()
                        S.uci_metadata = ds.metadata
                        S.uci_variables = ds.variables
                        S.image_shape = None
                    st.success(f"✓ Loaded **{ds.metadata.name}** — "
                               f"{S.df.shape[0]:,} rows × {S.df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Could not fetch dataset: {e}")

    if S.df is not None:
        df = S.df
        st.divider()
        t1, t2, t3, t4 = st.tabs(["Preview", "Column info", "Missing values", "Dataset Metadata"])
        with t1:
            if S.image_shape:
                h, w = S.image_shape
                pix_cols = [c for c in df.columns if c.startswith("pixel_")]
                if len(pix_cols) == h * w:
                    st.caption(f"Each row is a flattened {h}×{w} grayscale image "
                               f"({h*w} pixel features, intensity 0–16). Sample images:")
                    n_show = min(10, len(df))
                    titles = ([f"label={int(df.iloc[i]['target'])}" for i in range(n_show)]
                               if "target" in df.columns else None)
                    fig_img = make_subplots(rows=1, cols=n_show, horizontal_spacing=0.02,
                                             subplot_titles=titles)
                    for i in range(n_show):
                        img = df.iloc[i][pix_cols].values.astype(float).reshape(h, w)
                        fig_img.add_trace(
                            go.Heatmap(z=img, colorscale="Greys", showscale=False),
                            row=1, col=i + 1)
                    fig_img.update_xaxes(showticklabels=False, visible=False)
                    fig_img.update_yaxes(showticklabels=False, visible=False, autorange="reversed")
                    fig_img.update_layout(height=170, margin=dict(t=24, b=0, l=0, r=0))
                    st.plotly_chart(fig_img, use_container_width=True)
                    st.divider()
            st.dataframe(df.head(10), use_container_width=True)
        with t2:
            info = pd.DataFrame({
                "dtype":   df.dtypes.astype(str),
                "# unique": df.nunique(),
                "sample values": [str(df[c].dropna().unique()[:4].tolist()) for c in df.columns],
            })
            st.dataframe(info, use_container_width=True)
        with t3:
            miss = df.isnull().sum()
            miss = miss[miss > 0]
            if miss.empty:
                st.success("No missing values.")
            else:
                st.warning(f"{len(miss)} column(s) have missing values — auto-filled on next step.")
                mdf = pd.DataFrame({"missing": miss.values,
                                     "%": (miss.values/len(df)*100).round(1)}, index=miss.index)
                st.dataframe(mdf, use_container_width=True)
        with t4:
            if S.uci_metadata is not None:
                meta, vars_df = S.uci_metadata, S.uci_variables
                st.markdown(f"### {meta.name}")
                doi_txt = f"  ·  DOI: `{meta.dataset_doi}`" if meta.dataset_doi else ""
                st.caption(f"Source: [UCI Machine Learning Repository]"
                           f"({meta.repository_url}){doi_txt}")
                if meta.abstract:
                    st.markdown(f"*{meta.abstract}*")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Instances", meta.num_instances or df.shape[0])
                mc2.metric("Features", meta.num_features or (df.shape[1] - 1))
                mc3.metric("Missing values",
                           "Yes" if str(meta.has_missing_values).lower() == "yes" else "No")
                mc4.metric("Year created", meta.year_of_dataset_creation or "—")

                fact_rows = {
                    "Subject area":         meta.area,
                    "Task(s)":              ", ".join(meta.tasks) if meta.tasks else None,
                    "Data characteristics": ", ".join(meta.characteristics) if meta.characteristics else None,
                    "Feature types":        ", ".join(meta.feature_types) if meta.feature_types else None,
                    "Last updated":         meta.last_updated,
                }
                fact_rows = {k: v for k, v in fact_rows.items() if v}
                if fact_rows:
                    st.table(pd.DataFrame(fact_rows.items(), columns=["Field", "Value"]).set_index("Field"))

                ai = meta.additional_info or {}
                desc_rows = {
                    "Summary":                  ai.get("summary"),
                    "Purpose":                  ai.get("purpose"),
                    "Instances represent":      ai.get("instances_represent"),
                    "Recommended data splits":  ai.get("recommended_data_splits"),
                    "Sensitive data":           ai.get("sensitive_data"),
                    "Preprocessing":            ai.get("preprocessing_description"),
                }
                desc_rows = {k: v for k, v in desc_rows.items()
                             if v and str(v).strip().upper() != "N/A"}
                if desc_rows:
                    with st.expander("Full dataset description", expanded=False):
                        for k, v in desc_rows.items():
                            st.markdown(f"**{k}:** {v}")

                if vars_df is not None and not vars_df.empty:
                    st.markdown("**Variable dictionary** (as published by UCI)")
                    st.dataframe(vars_df.fillna("—"), use_container_width=True)
            else:
                st.caption("Auto-generated profile. Connect a dataset from the "
                           "**UCI ML Repository** above for full official metadata "
                           "(abstract, task, variable dictionary, citation, etc.).")
                n_num = int(sum(is_numeric(df[c]) for c in df.columns))
                n_cat = df.shape[1] - n_num
                n_dup = int(df.duplicated().sum())
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Rows", f"{df.shape[0]:,}")
                mc2.metric("Columns", df.shape[1])
                mc3.metric("Numeric / Categorical", f"{n_num} / {n_cat}")
                mc4.metric("Duplicate rows", n_dup)
                st.dataframe(df.describe(include="all").T.fillna("—"), use_container_width=True)

        st.button("Next →", on_click=advance, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Configure Columns
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 1:
    st.title("Step 2 — Configure Columns")
    if S.df is None:
        st.error("No data loaded. Go back."); st.stop()

    # Fill missing values
    df = S.df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median() if is_numeric(df[col]) else df[col].mode()[0])
    S.df = df

    # ── Target + task ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        default_target_idx = len(df.columns) - 1
        target_col = st.selectbox("🎯 Target column (what to predict)",
                                   df.columns.tolist(), index=default_target_idx)
    with c2:
        auto_task = detect_task(df[target_col])
        task = st.selectbox("Task type",
                             ["regression", "binary", "multiclass"],
                             index=["regression","binary","multiclass"].index(auto_task),
                             help="Auto-detected from target column. Override if needed.")

    S.target_col = target_col
    S.task = task

    task_desc = {"regression": "📈 Predict a continuous number",
                 "binary": "⚖️ Predict one of two classes",
                 "multiclass": "🎯 Predict one of 3+ classes"}
    st.caption(task_desc[task] + f"  ·  Target unique values: {df[target_col].nunique()}")

    # ── Feature selection ──────────────────────────────────────────────────
    st.divider()
    default_feats = [c for c in df.columns if c != target_col]
    feature_cols = st.multiselect("Feature columns (inputs)",
                                   default_feats, default=default_feats)
    if not feature_cols:
        st.warning("Select at least one feature column."); st.stop()
    S.feature_cols = feature_cols

    # ── Encoding table ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Column Encoding")
    st.caption("Edit the table below. For **Ordinal** encoding, fill in the category order (low → high). "
               "**Text (TF-IDF)** turns a free-text column into up to 200 word-frequency features.")

    # Build or restore config DataFrame
    if (S.col_config_df is None
            or set(S.col_config_df["Column"]) != set(feature_cols)):
        S.col_config_df = pd.DataFrame({
            "Column": feature_cols,
            "Detected dtype": [detected_dtype(c, df) for c in feature_cols],
            "Encoding": [default_encoding(c, df) for c in feature_cols],
            "Ordinal order (low → high)": ["" for _ in feature_cols],
        })

    # Align to current feature_cols (handles adds/removals)
    existing = S.col_config_df.set_index("Column")
    rows = []
    for c in feature_cols:
        if c in existing.index:
            rows.append(existing.loc[c].tolist() + [])  # keep existing
        else:
            rows.append([detected_dtype(c, df), default_encoding(c, df), ""])
    col_config_df = pd.DataFrame(rows, columns=["Detected dtype","Encoding","Ordinal order (low → high)"])
    col_config_df.insert(0, "Column", feature_cols)

    enc_options_num  = ["StandardScaler", "MinMaxScaler", "None (keep)", "Drop"]
    enc_options_cat  = ["One-Hot", "Ordinal", "Drop"]
    enc_options_text = ["Text (TF-IDF)", "One-Hot", "Drop"]
    all_enc_options = list(dict.fromkeys(enc_options_num + enc_options_cat + enc_options_text))

    edited = st.data_editor(
        col_config_df,
        column_config={
            "Column":        st.column_config.TextColumn(disabled=True, width="small"),
            "Detected dtype":st.column_config.TextColumn(disabled=True, width="small"),
            "Encoding":      st.column_config.SelectboxColumn(
                                 "Encoding", options=all_enc_options, width="medium"),
            "Ordinal order (low → high)": st.column_config.TextColumn(
                                 "Ordinal order (low → high, comma-separated)",
                                 width="large",
                                 help="Only used when Encoding = Ordinal. E.g.: Poor, Good, Excellent"),
        },
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="col_config_editor",
    )
    S.col_config_df = edited

    active = edited[edited["Encoding"] != "Drop"]
    if active.empty:
        st.error("All columns are set to Drop. Keep at least one."); st.stop()
    st.success(f"**{len(active)}** feature columns will be used after encoding.")

    st.button("Next →", on_click=advance, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Split Data
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 2:
    st.title("Step 3 — Train / Test Split")
    if S.df is None or S.col_config_df is None:
        st.error("Complete earlier steps first."); st.stop()

    df = S.df
    target_col  = S.target_col
    feature_cols = [c for c in S.feature_cols
                    if c in S.col_config_df.set_index("Column").index
                    and S.col_config_df.set_index("Column").loc[c,"Encoding"] != "Drop"]

    c1, c2, c3 = st.columns(3)
    test_pct = c1.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
    seed     = int(c2.number_input("Random seed", 0, 9999, 42))
    stratify = False
    if S.task in ("binary","multiclass"):
        stratify = c3.checkbox("Stratify split", value=True,
                               help="Keeps class proportions equal in both splits.")

    if st.button("Apply split & fit preprocessor", type="primary"):
        X = df[feature_cols]
        y = df[target_col]
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_pct, random_state=seed,
                stratify=(y if stratify else None)
            )
            pre = build_preprocessor(df, feature_cols, S.col_config_df)
            pre.fit(X_tr)
            X_tr_t = pre.transform(X_tr)
            X_te_t = pre.transform(X_te)
            try:   fn = list(pre.get_feature_names_out())
            except: fn = [f"f{i}" for i in range(X_tr_t.shape[1])]

            S.preprocessor    = pre
            S.feat_names_out  = fn
            S.X_tr, S.X_te    = X_tr_t, X_te_t
            S.y_tr, S.y_te    = y_tr.values, y_te.values
            S.X_tr_raw = X_tr.reset_index(drop=True)
            S.X_te_raw = X_te.reset_index(drop=True)
            # reset downstream
            S.model = None; S.y_pred = None; S.y_prob = None
            st.success("Split applied and preprocessor fitted!")
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

    if S.X_tr is not None:
        metric_row({
            "Train samples":           S.X_tr.shape[0],
            "Test samples":            S.X_te.shape[0],
            "Features after encoding": S.X_tr.shape[1],
        })

        t_stats, t_enc, t_bal = st.tabs(["Split Stats", "Encoded Preview", "Feature Balance"])

        with t_stats:
            if S.task in ("binary","multiclass"):
                classes, tr_c = np.unique(S.y_tr, return_counts=True)
                _,       te_c = np.unique(S.y_te, return_counts=True)
                cdf = pd.DataFrame({
                    "Class":   classes.astype(str),
                    "Train #": tr_c, "Train %": (tr_c/len(S.y_tr)*100).round(1),
                    "Test #":  te_c, "Test %":  (te_c/len(S.y_te)*100).round(1),
                })
                st.dataframe(cdf.set_index("Class"), use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                fig_tr = px.histogram(x=S.y_tr, nbins=40, title="Train — target distribution",
                                      labels={"x": S.target_col},
                                      color_discrete_sequence=["steelblue"])
                fig_te = px.histogram(x=S.y_te, nbins=40, title="Test — target distribution",
                                      labels={"x": S.target_col},
                                      color_discrete_sequence=["coral"])
                for f in (fig_tr, fig_te):
                    f.update_layout(height=280, margin=dict(t=36,b=0,l=0,r=0))
                c1.plotly_chart(fig_tr, use_container_width=True)
                c2.plotly_chart(fig_te, use_container_width=True)

        with t_enc:
            fn = S.feat_names_out
            preview = pd.DataFrame(S.X_tr[:8], columns=fn).round(4)
            st.caption("First 8 rows of the **training** feature matrix after encoding & scaling.")
            st.dataframe(preview, use_container_width=True)
            stats = pd.DataFrame(S.X_tr, columns=fn).describe().round(4)
            with st.expander("Column statistics (mean / std / min / max …)"):
                st.dataframe(stats, use_container_width=True)

        with t_bal:
            raw_tr = S.X_tr_raw
            raw_te = S.X_te_raw
            num_cols_bal = [c for c in raw_tr.columns if is_numeric(raw_tr[c])][:4]
            if num_cols_bal:
                cols_bal = st.columns(len(num_cols_bal))
                for ax, col in zip(cols_bal, num_cols_bal):
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Histogram(x=raw_tr[col], name="Train",
                                                  marker_color="steelblue", opacity=0.65))
                    fig_b.add_trace(go.Histogram(x=raw_te[col], name="Test",
                                                  marker_color="coral", opacity=0.65))
                    fig_b.update_layout(barmode="overlay", title=col,
                                         height=220, showlegend=(col==num_cols_bal[0]),
                                         margin=dict(t=30,b=0,l=0,r=0))
                    ax.plotly_chart(fig_b, use_container_width=True)
            if S.task in ("binary","multiclass"):
                classes, tr_c = np.unique(S.y_tr, return_counts=True)
                _,       te_c = np.unique(S.y_te, return_counts=True)
                fig_cls = go.Figure([
                    go.Bar(name="Train", x=classes.astype(str), y=tr_c,
                           marker_color="steelblue"),
                    go.Bar(name="Test",  x=classes.astype(str), y=te_c,
                           marker_color="coral"),
                ])
                fig_cls.update_layout(barmode="group",
                                       title="Class Balance — Train vs Test", height=300)
                st.plotly_chart(fig_cls, use_container_width=True)

        st.button("Next →", on_click=advance, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train & Tune
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 3:
    st.title("Step 4 — Train & Hyperparameter Tuning")
    if S.X_tr is None:
        st.error("Complete the Split step first."); st.stop()

    # ── Explore Training Data ──────────────────────────────────────────────
    with st.expander("📊 Explore Training Data", expanded=False):
        raw_tr = S.X_tr_raw
        num_cols_exp = [c for c in raw_tr.columns if is_numeric(raw_tr[c])][:6]
        t_corr, t_scatter, t_tdist = st.tabs(["Correlation Heatmap", "Scatter Matrix", "Target Distribution"])

        with t_corr:
            if len(num_cols_exp) >= 2:
                corr_df = raw_tr[num_cols_exp].corr().round(2)
                fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu",
                                     color_continuous_midpoint=0,
                                     title="Feature Correlation (training set — numeric features)")
                fig_corr.update_layout(height=420)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Need at least 2 numeric features for a correlation heatmap.")

        with t_scatter:
            plot_cols = num_cols_exp[:5]
            if len(plot_cols) >= 2:
                sc_df = raw_tr[plot_cols].copy()
                sc_df[S.target_col] = S.y_tr
                if len(sc_df) > 500:
                    sc_df = sc_df.sample(500, random_state=42)
                fig_sc = px.scatter_matrix(
                    sc_df, dimensions=plot_cols, color=S.target_col,
                    opacity=0.5,
                    title=f"Scatter Matrix — training data (≤500 samples, ≤5 numeric features)",
                )
                fig_sc.update_traces(diagonal_visible=False, showupperhalf=False)
                fig_sc.update_layout(height=520)
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("Need at least 2 numeric features for a scatter matrix.")

        with t_tdist:
            if S.task == "regression":
                fig_td = px.histogram(x=S.y_tr, nbins=40,
                                      title=f"Training Target Distribution — {S.target_col}",
                                      labels={"x": S.target_col},
                                      color_discrete_sequence=["steelblue"])
                fig_td.update_layout(height=320)
                st.plotly_chart(fig_td, use_container_width=True)
            else:
                cl, ct = np.unique(S.y_tr, return_counts=True)
                fig_td = px.bar(x=cl.astype(str), y=ct,
                                title=f"Class Distribution (training) — {S.target_col}",
                                labels={"x": "Class", "y": "Count"},
                                color=cl.astype(str),
                                color_discrete_sequence=px.colors.qualitative.Set2)
                fig_td.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_td, use_container_width=True)

    # ── Model selector ─────────────────────────────────────────────────────
    if S.task == "regression":
        model_opts = ["Linear Regression", "Ridge (L2)", "Lasso (L1)", "ElasticNet",
                      "Multilayer Perceptron (MLP)"]
    else:
        model_opts = ["Logistic Regression — L2", "Logistic Regression — L1",
                      "Logistic Regression — no regularization",
                      "Multilayer Perceptron (MLP)"]
    model_name = st.selectbox("Model", model_opts)

    # ── Hyperparameter grid ────────────────────────────────────────────────
    st.subheader("Hyperparameter Search Grid")
    ALPHA_OPTS     = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    C_OPTS         = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    L1_OPTS        = [0.1, 0.2, 0.5, 0.8, 0.9]
    MLP_ALPHA_OPTS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    param_grid = {}
    if model_name == "Linear Regression":
        estimator = LinearRegression()
        st.info("Linear Regression has no regularization — it will be fitted directly.")

    elif model_name == "Ridge (L2)":
        alphas = st.multiselect("α (lambda) values to try", ALPHA_OPTS,
                                 default=[0.01, 0.1, 1.0, 10.0, 100.0])
        estimator  = Ridge()
        param_grid = {"alpha": alphas}

    elif model_name == "Lasso (L1)":
        alphas = st.multiselect("α (lambda) values to try", ALPHA_OPTS,
                                 default=[0.001, 0.01, 0.1, 1.0])
        estimator  = Lasso(max_iter=10_000)
        param_grid = {"alpha": alphas}

    elif model_name == "ElasticNet":
        c1, c2 = st.columns(2)
        alphas    = c1.multiselect("α values", ALPHA_OPTS, default=[0.001, 0.01, 0.1, 1.0])
        l1_ratios = c2.multiselect("l1_ratio  (0=L2 · 1=L1)", L1_OPTS, default=[0.2, 0.5, 0.8])
        estimator  = ElasticNet(max_iter=10_000)
        param_grid = {"alpha": alphas, "l1_ratio": l1_ratios}

    elif model_name == "Logistic Regression — L2":
        c_vals     = st.multiselect("C = 1/λ  (smaller = stronger reg)", C_OPTS,
                                     default=[0.01, 0.1, 1.0, 10.0])
        mc = "multinomial" if S.task == "multiclass" else "auto"
        estimator  = LogisticRegression(penalty="l2", solver="lbfgs",
                                        multi_class=mc, max_iter=2000)
        param_grid = {"C": c_vals}

    elif model_name == "Logistic Regression — L1":
        c_vals     = st.multiselect("C = 1/λ", C_OPTS, default=[0.01, 0.1, 1.0, 10.0])
        estimator  = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000)
        param_grid = {"C": c_vals}

    elif model_name == "Multilayer Perceptron (MLP)":
        st.markdown("**Design your network**")

        backend_opts = ["scikit-learn"] + (["PyTorch"] if TORCH_AVAILABLE else [])
        if not TORCH_AVAILABLE:
            st.caption("Install `torch` (`pip install torch`) to enable the PyTorch backend.")
        backend = st.radio("Backend", backend_opts, horizontal=True)

        n_layers = st.slider("Hidden layers", 1, 4, 2)
        layer_defaults = [64, 32, 16, 8]
        layer_cols = st.columns(n_layers)
        hidden_layer_sizes = [
            int(c.number_input(f"Neurons — hidden {i+1}", min_value=1, max_value=256,
                                value=layer_defaults[i], step=1, key=f"mlp_layer_{i}"))
            for i, c in enumerate(layer_cols)
        ]
        c1, c2 = st.columns(2)
        activation = c1.selectbox("Activation", ["relu", "tanh", "logistic"])
        solver_opts = ["adam", "sgd", "lbfgs"] if backend == "scikit-learn" else ["adam", "sgd"]
        solver     = c2.selectbox("Solver", solver_opts)

        n_in  = S.X_tr.shape[1]
        n_out = 1 if S.task in ("regression", "binary") else len(np.unique(S.y_tr))
        st.plotly_chart(mlp_diagram([n_in] + hidden_layer_sizes + [n_out]),
                         use_container_width=True)

        alphas = st.multiselect("α (L2 regularization) values to try", MLP_ALPHA_OPTS,
                                 default=[1e-4, 1e-3, 1e-2])

        if backend == "PyTorch":
            c3, c4 = st.columns(2)
            learning_rate_init = c3.number_input("Learning rate", min_value=1e-5, max_value=1.0,
                                                   value=1e-3, step=1e-4, format="%.5f")
            batch_size = int(c4.number_input("Batch size", min_value=1, max_value=512,
                                              value=32, step=1))
            max_iter = st.slider("Epochs", 10, 1000, 300, 10)
            mlp_kwargs = dict(hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation,
                               solver=solver, learning_rate_init=learning_rate_init,
                               batch_size=batch_size, max_iter=max_iter, random_state=42)
            estimator  = (TorchMLPRegressor(**mlp_kwargs) if S.task == "regression"
                          else TorchMLPClassifier(**mlp_kwargs))
        else:
            mlp_kwargs = dict(hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation,
                               solver=solver, max_iter=300, random_state=42)
            estimator  = (MLPRegressor(**mlp_kwargs) if S.task == "regression"
                          else MLPClassifier(**mlp_kwargs))
        param_grid = {"alpha": alphas}

    else:
        mc = "multinomial" if S.task == "multiclass" else "auto"
        estimator  = LogisticRegression(penalty=None, solver="lbfgs",
                                        multi_class=mc, max_iter=2000)
        param_grid = {}

    # ── CV settings ────────────────────────────────────────────────────────
    st.subheader("Cross-Validation Settings")
    c1, c2 = st.columns(2)
    k_folds = c1.select_slider("K folds", [3, 5, 10], value=5)

    if S.task == "regression":
        score_opts = {"R²": "r2",
                      "Neg MSE": "neg_mean_squared_error",
                      "Neg MAE": "neg_mean_absolute_error"}
    elif S.task == "binary":
        score_opts = {"AUC-ROC": "roc_auc",
                      "Accuracy": "accuracy",
                      "F1": "f1"}
    else:
        score_opts = {"Accuracy": "accuracy",
                      "F1 (macro)": "f1_macro"}
    score_label = c2.selectbox("Scoring metric", list(score_opts))
    scoring     = score_opts[score_label]

    # ── Run ────────────────────────────────────────────────────────────────
    if st.button("▶  Run GridSearchCV", type="primary",
                  disabled=(bool(param_grid) and any(len(v)==0 for v in param_grid.values()))):
        cv = (StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
              if S.task in ("binary","multiclass")
              else KFold(n_splits=k_folds, shuffle=True, random_state=42))
        with st.spinner("Running cross-validation …"):
            try:
                if param_grid:
                    is_torch_est = isinstance(estimator, (TorchMLPClassifier, TorchMLPRegressor))
                    gs = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                                      return_train_score=True, n_jobs=(1 if is_torch_est else -1))
                    gs.fit(S.X_tr, S.y_tr)
                    S.model        = gs.best_estimator_
                    S.best_params  = gs.best_params_
                    S.cv_results_df = pd.DataFrame(gs.cv_results_)
                else:
                    estimator.fit(S.X_tr, S.y_tr)
                    S.model        = estimator
                    S.best_params  = {}
                    S.cv_results_df = None
                S.y_pred = None; S.y_prob = None
                st.success("Done!")
            except Exception as e:
                st.error(f"Training failed: {e}"); st.stop()

    # ── Results ────────────────────────────────────────────────────────────
    if S.model is not None:
        if S.best_params:
            st.markdown("**Best hyperparameters:** " +
                        "  ·  ".join(f"`{k}={v}`" for k,v in S.best_params.items()))

        cv_df = S.cv_results_df
        if cv_df is not None:
            param_keys = [c for c in cv_df.columns if c.startswith("param_")]

            if len(param_keys) == 1:
                pk = param_keys[0]
                pname = pk.replace("param_","")
                xs = cv_df[pk].astype(float)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=cv_df["mean_train_score"].astype(float),
                                          mode="lines+markers", name="Train",
                                          line=dict(color="steelblue")))
                fig.add_trace(go.Scatter(x=xs, y=cv_df["mean_test_score"].astype(float),
                                          mode="lines+markers", name=f"Val ({score_label})",
                                          line=dict(color="coral")))
                val = cv_df["mean_test_score"].astype(float)
                std = cv_df["std_test_score"].astype(float)
                fig.add_trace(go.Scatter(
                    x=list(xs)+list(xs[::-1]),
                    y=list(val+std)+list((val-std)[::-1]),
                    fill="toself", fillcolor="rgba(220,80,80,0.12)",
                    line=dict(width=0), showlegend=False))
                best_x = S.best_params.get(pname)
                if best_x is not None:
                    fig.add_vline(x=float(best_x), line_color="green", line_dash="dash",
                                   annotation_text=f"Best={best_x}")
                use_log = (xs.max()/(xs.min()+1e-9)) > 50
                fig.update_layout(title=f"CV Results — {score_label}",
                                   xaxis_title=pname, yaxis_title="Score",
                                   xaxis_type="log" if use_log else "linear", height=380)
                st.plotly_chart(fig, use_container_width=True)

            elif len(param_keys) == 2:
                pk1, pk2 = param_keys
                pivot = cv_df.pivot_table("mean_test_score", index=pk2, columns=pk1, aggfunc="mean")
                fig = px.imshow(pivot.round(4), text_auto=True, color_continuous_scale="RdYlGn",
                                 title=f"CV {score_label} ({pk1.replace('param_','')} × {pk2.replace('param_','')})")
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)

        st.button("Next →", on_click=advance, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 4:
    st.title("Step 5 — Evaluate")
    if S.model is None:
        st.error("Train a model first."); st.stop()

    mdl = S.model
    X_tr, X_te = S.X_tr, S.X_te
    y_tr, y_te = S.y_tr, S.y_te
    is_mlp = isinstance(mdl, (MLPClassifier, MLPRegressor, TorchMLPClassifier, TorchMLPRegressor))
    coef_tab_label = "Network" if is_mlp else "Coefficients"

    def render_model_tab():
        if is_mlp:
            st.plotly_chart(
                mlp_diagram([X_tr.shape[1]] + list(mdl.hidden_layer_sizes) + [mdl.n_outputs_]),
                use_container_width=True)
            if hasattr(mdl, "loss_curve_"):
                fig_loss = go.Figure([go.Scatter(y=mdl.loss_curve_, mode="lines",
                                                  line=dict(color="steelblue"))])
                fig_loss.update_layout(title="Training Loss Curve", xaxis_title="Iteration",
                                        yaxis_title="Loss", height=320)
                st.plotly_chart(fig_loss, use_container_width=True)
            else:
                st.info(f"Loss curve not available for solver='{mdl.solver}' "
                        "(only 'sgd'/'adam' expose it).")
            return True
        return False

    y_pred_te = mdl.predict(X_te)
    y_pred_tr = mdl.predict(X_tr)
    y_prob_te = None
    if S.task != "regression" and hasattr(mdl,"predict_proba"):
        y_prob_te = mdl.predict_proba(X_te)
    S.y_pred = y_pred_te
    S.y_prob = y_prob_te

    # ── Regression ──────────────────────────────────────────────────────────
    if S.task == "regression":
        t_met, t_pred, t_qq, t_cross, t_coef, t_samp = st.tabs(
            ["Metrics", "Predictions", "Residual Q-Q", "Cross-plot", coef_tab_label, "Sample Predictions"])

        with t_met:
            metric_row({
                "Train R²":  r2_score(y_tr, y_pred_tr),
                "Test R²":   r2_score(y_te, y_pred_te),
                "Test RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred_te))),
                "Test MAE":  mean_absolute_error(y_te, y_pred_te),
                "Overfit":   float(r2_score(y_tr,y_pred_tr)-r2_score(y_te,y_pred_te)),
            })

        with t_pred:
            c1, c2 = st.columns(2)
            lim = [float(y_te.min()), float(y_te.max())]
            fig_pva = go.Figure([
                go.Scatter(x=y_te, y=y_pred_te, mode="markers",
                           marker=dict(color="steelblue",opacity=0.3,size=5), name="Test"),
                go.Scatter(x=lim, y=lim, mode="lines",
                           line=dict(color="red",dash="dash"), name="Perfect"),
            ])
            fig_pva.update_layout(title="Predicted vs Actual",
                                   xaxis_title="Actual", yaxis_title="Predicted", height=400)
            c1.plotly_chart(fig_pva, use_container_width=True)

            resid = y_te - y_pred_te
            fig_res = px.histogram(x=resid, nbins=50, title="Residuals",
                                    labels={"x":"Actual − Predicted"},
                                    color_discrete_sequence=["steelblue"])
            fig_res.add_vline(x=0, line_color="red", line_width=2)
            fig_res.update_layout(height=400)
            c2.plotly_chart(fig_res, use_container_width=True)

        with t_qq:
            resid_qq = (y_te - y_pred_te).astype(float)
            (theo_q, ord_resid), (slope, intercept, r) = scipy_stats.probplot(resid_qq, dist="norm")
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=theo_q, y=ord_resid, mode="markers",
                marker=dict(color="steelblue", size=6, opacity=0.65),
                name="Residuals",
                text=[f"Theoretical: {t:.3f}<br>Residual: {o:.3f}" for t, o in zip(theo_q, ord_resid)],
                hoverinfo="text",
            ))
            line_x = np.array([theo_q.min(), theo_q.max()])
            fig_qq.add_trace(go.Scatter(
                x=line_x, y=slope * line_x + intercept, mode="lines",
                line=dict(color="red", dash="dash", width=2), name="Normal fit",
            ))
            fig_qq.update_layout(
                title=f"Residual Q-Q Plot  (R² of fit = {r**2:.4f})",
                xaxis_title="Theoretical quantiles (Normal)",
                yaxis_title="Ordered residuals",
                height=460,
            )
            st.plotly_chart(fig_qq, use_container_width=True)
            st.caption(
                "Points hugging the dashed line indicate residuals are approximately "
                "normally distributed. Curvature at the tails suggests skew or heavy "
                "tails; an S-shape suggests skewed residuals — useful for checking OLS "
                "inference assumptions (not required for prediction accuracy alone)."
            )

        with t_cross:
            abs_err = np.abs(y_te - y_pred_te)
            lim_c = [float(min(y_te.min(), y_pred_te.min()) * 0.98),
                     float(max(y_te.max(), y_pred_te.max()) * 1.02)]
            max_ae = abs_err.max() + 1e-9
            marker_colors = [
                f"rgb({int(255 * e/max_ae)},{int(200*(1-e/max_ae))},80)"
                for e in abs_err
            ]
            fig_cross = go.Figure()
            fig_cross.add_trace(go.Scatter(
                x=y_te, y=y_pred_te,
                mode="markers",
                marker=dict(color=marker_colors, size=7, opacity=0.75,
                            line=dict(width=0.4, color="white")),
                text=[f"Actual: {a:.3f}<br>Predicted: {p:.3f}<br>|Error|: {e:.3f}"
                      for a, p, e in zip(y_te, y_pred_te, abs_err)],
                hoverinfo="text",
                name="Test samples",
            ))
            fig_cross.add_trace(go.Scatter(
                x=lim_c, y=lim_c,
                mode="lines",
                line=dict(color="black", dash="dash", width=1.5),
                name="Perfect fit (y = x)",
            ))
            fig_cross.update_layout(
                title=f"Cross-plot — Predicted vs Actual ({S.target_col})",
                xaxis_title=f"Actual  [{S.target_col}]",
                yaxis_title=f"Predicted  [{S.target_col}]",
                height=520,
                legend=dict(orientation="h", y=-0.12),
            )
            st.plotly_chart(fig_cross, use_container_width=True)
            r2_te = r2_score(y_te, y_pred_te)
            rmse_te = float(np.sqrt(mean_squared_error(y_te, y_pred_te)))
            st.caption(
                f"Test R² = **{r2_te:.4f}**  ·  RMSE = **{rmse_te:.4f}**  ·  "
                f"Points on the dashed line = perfect predictions  ·  "
                f"Color: green = small error → red = large error"
            )

        with t_coef:
            if not render_model_tab() and hasattr(mdl,"coef_"):
                coef = mdl.coef_.flatten()
                fn = S.feat_names_out[:len(coef)] or [f"f{i}" for i in range(len(coef))]
                cdf = pd.DataFrame({"Feature":fn,"Coefficient":coef})
                cdf = cdf.sort_values("Coefficient",key=abs,ascending=True)
                fig_c = px.bar(cdf, x="Coefficient", y="Feature", orientation="h",
                                color="Coefficient", color_continuous_scale="RdBu",
                                color_continuous_midpoint=0, title="Learned Weights")
                fig_c.update_layout(height=max(350,len(coef)*22))
                st.plotly_chart(fig_c, use_container_width=True)
                nz = int(np.sum(np.abs(coef) < 1e-8))
                if nz: st.info(f"{nz} weight(s) are exactly zero → Lasso feature selection.")

        with t_samp:
            n_show = st.slider("Rows to show", 10, min(100, len(y_te)), 20, 5,
                               key="n_show_reg")
            sort_by = st.radio("Sort by", ["Random", "Worst errors first"],
                               horizontal=True, key="sort_reg")
            samp_df = pd.DataFrame({
                "Actual":    y_te.round(4),
                "Predicted": y_pred_te.round(4),
                "Residual":  (y_te - y_pred_te).round(4),
                "|Error|":   np.abs(y_te - y_pred_te).round(4),
            })
            if sort_by == "Worst errors first":
                samp_df = samp_df.sort_values("|Error|", ascending=False).head(n_show)
            else:
                samp_df = samp_df.sample(min(n_show, len(samp_df)), random_state=42)
            samp_df.index = range(1, len(samp_df) + 1)
            max_err = samp_df["|Error|"].max() + 1e-9
            def _err_color(val):
                t = val / max_err
                g = int(255 * (1 - t * 0.85))
                return f"background-color: rgba(255,{g},{g},0.55)"
            st.dataframe(
                samp_df.style.map(_err_color, subset=["|Error|"]),
                use_container_width=True,
            )

    # ── Classification ──────────────────────────────────────────────────────
    else:
        t_met, t_cm, t_roc, t_cross, t_thresh, t_coef, t_samp = st.tabs(
            ["Metrics","Confusion Matrix","ROC Curve","Cross-plot",
             "Threshold", coef_tab_label, "Sample Predictions"])
        avg  = "binary" if S.task == "binary" else "macro"
        classes = mdl.classes_

        with t_met:
            mets = {
                "Accuracy":  accuracy_score(y_te, y_pred_te),
                "Precision": precision_score(y_te, y_pred_te, average=avg, zero_division=0),
                "Recall":    recall_score(y_te, y_pred_te, average=avg, zero_division=0),
                "F1":        f1_score(y_te, y_pred_te, average=avg, zero_division=0),
            }
            if y_prob_te is not None and S.task == "binary":
                mets["AUC-ROC"] = roc_auc_score(y_te, y_prob_te[:,1])
            metric_row(mets)

            # Probability histogram (binary)
            if y_prob_te is not None and S.task == "binary":
                fig_ph = go.Figure()
                for lbl,color,name in [(0,"steelblue","Negative"),(1,"crimson","Positive")]:
                    fig_ph.add_trace(go.Histogram(
                        x=y_prob_te[y_te==lbl,1], nbinsx=30,
                        name=name, marker_color=color, opacity=0.65))
                fig_ph.update_layout(barmode="overlay",
                                      title="Predicted Probability Distribution",
                                      xaxis_title="P(Positive class)",
                                      height=320)
                st.plotly_chart(fig_ph, use_container_width=True)

        with t_cm:
            cm = confusion_matrix(y_te, y_pred_te, labels=classes)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                x=[f"Pred {c}" for c in classes],
                                y=[f"True {c}" for c in classes],
                                title="Confusion Matrix — default threshold 0.5")
            fig_cm.update_layout(coloraxis_showscale=False, height=420)
            st.plotly_chart(fig_cm, use_container_width=True)

        with t_roc:
            if y_prob_te is not None:
                fig_roc = go.Figure()
                if S.task == "binary":
                    fpr, tpr, _ = roc_curve(y_te, y_prob_te[:,1])
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                                  name=f"AUC={auc(fpr,tpr):.4f}",
                                                  line=dict(color="steelblue",width=2.5)))
                else:
                    pal = px.colors.qualitative.Set1
                    for i, cls in enumerate(np.unique(y_te)):
                        y_bin = (y_te==cls).astype(int)
                        fpr, tpr, _ = roc_curve(y_bin, y_prob_te[:,i])
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                                      name=f"class {cls}  AUC={auc(fpr,tpr):.3f}",
                                                      line=dict(color=pal[i%len(pal)],width=2)))
                fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                              line=dict(color="gray",dash="dash"),name="Random"))
                fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                                       title="ROC Curve", height=430)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("predict_proba not available for this model configuration.")

        with t_cross:
            correct = y_te == y_pred_te
            rng = np.random.RandomState(42)
            if S.task == "binary" and y_prob_te is not None:
                jitter = rng.uniform(-0.18, 0.18, len(y_te))
                fig_cross = go.Figure()
                for flag, label, color in [(True,"Correct","steelblue"),(False,"Incorrect","crimson")]:
                    mask = correct == flag
                    fig_cross.add_trace(go.Scatter(
                        x=y_te[mask].astype(float) + jitter[mask],
                        y=y_prob_te[mask, 1],
                        mode="markers",
                        marker=dict(color=color, opacity=0.55, size=6),
                        name=label,
                        text=[f"True: {t}<br>P(positive): {p:.3f}<br>Predicted: {pr}"
                              for t,p,pr in zip(y_te[mask], y_prob_te[mask,1], y_pred_te[mask])],
                        hoverinfo="text",
                    ))
                fig_cross.add_hline(y=0.5, line_dash="dash", line_color="gray",
                                    annotation_text="threshold = 0.5",
                                    annotation_position="top right")
                fig_cross.update_layout(
                    title="Cross-plot — Predicted Probability vs Actual Class",
                    xaxis=dict(tickvals=[0,1], ticktext=["Negative (0)","Positive (1)"],
                               title="Actual Class", range=[-0.5, 1.5]),
                    yaxis_title="Predicted Probability (positive class)",
                    height=480,
                )
                st.plotly_chart(fig_cross, use_container_width=True)
                st.caption("Each dot is one test sample. Blue = correctly classified, "
                           "red = misclassified. Horizontal jitter added to separate overlapping points.")

            elif S.task == "multiclass":
                class_list = list(mdl.classes_)
                c2i = {c: i for i, c in enumerate(class_list)}
                actual_idx = np.array([c2i[c] for c in y_te])
                pred_idx   = np.array([c2i[c] for c in y_pred_te])
                jx = rng.uniform(-0.22, 0.22, len(y_te))
                jy = rng.uniform(-0.22, 0.22, len(y_te))
                fig_cross = go.Figure()
                for flag, label, color in [(True,"Correct","steelblue"),(False,"Incorrect","crimson")]:
                    mask = correct == flag
                    fig_cross.add_trace(go.Scatter(
                        x=actual_idx[mask] + jx[mask],
                        y=pred_idx[mask]   + jy[mask],
                        mode="markers",
                        marker=dict(color=color, opacity=0.55, size=6),
                        name=label,
                        text=[f"True: {t}<br>Predicted: {p}"
                              for t,p in zip(y_te[mask], y_pred_te[mask])],
                        hoverinfo="text",
                    ))
                ticks = list(range(len(class_list)))
                labels = [str(c) for c in class_list]
                fig_cross.update_layout(
                    title="Cross-plot — Predicted Class vs Actual Class",
                    xaxis=dict(tickvals=ticks, ticktext=labels, title="Actual Class"),
                    yaxis=dict(tickvals=ticks, ticktext=labels, title="Predicted Class"),
                    height=480,
                )
                fig_cross.add_shape(type="line",
                    x0=-0.5, y0=-0.5, x1=len(class_list)-0.5, y1=len(class_list)-0.5,
                    line=dict(color="black", dash="dash", width=1.5))
                st.plotly_chart(fig_cross, use_container_width=True)
                st.caption("Dots on the dashed diagonal = correct predictions. "
                           "Jitter added to separate overlapping points.")
            else:
                st.info("Cross-plot requires predict_proba.")

        with t_thresh:
            if y_prob_te is not None and S.task == "binary":
                thresh = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
                y_thresh = (y_prob_te[:,1] >= thresh).astype(int)
                c1, c2 = st.columns(2)
                metric_row({
                    "Accuracy":  accuracy_score(y_te, y_thresh),
                    "Precision": precision_score(y_te, y_thresh, zero_division=0),
                    "Recall":    recall_score(y_te, y_thresh, zero_division=0),
                    "F1":        f1_score(y_te, y_thresh, zero_division=0),
                })
                cm_t = confusion_matrix(y_te, y_thresh, labels=classes)
                fig_t = px.imshow(cm_t, text_auto=True, color_continuous_scale="Oranges",
                                   x=[f"Pred {c}" for c in classes],
                                   y=[f"True {c}" for c in classes],
                                   title=f"Confusion Matrix at threshold={thresh}")
                fig_t.update_layout(coloraxis_showscale=False, height=380)
                st.plotly_chart(fig_t, use_container_width=True)
                tn,fp,fn,tp = cm_t.ravel()
                st.caption(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}  "
                            f"→  FPR={fp/(fp+tn+1e-9):.3f}  FNR={fn/(fn+tp+1e-9):.3f}")
            else:
                st.info("Threshold tuning is available for binary classification with predict_proba.")

        with t_coef:
            if not render_model_tab() and hasattr(mdl,"coef_"):
                coef = mdl.coef_
                fn   = S.feat_names_out or [f"f{i}" for i in range(coef.shape[1])]
                if coef.shape[0] == 1:
                    cdf = pd.DataFrame({"Feature":fn[:coef.shape[1]],"Coefficient":coef[0]})
                    cdf = cdf.sort_values("Coefficient",key=abs,ascending=True)
                    fig_c = px.bar(cdf, x="Coefficient",y="Feature",orientation="h",
                                    color="Coefficient",color_continuous_scale="RdBu",
                                    color_continuous_midpoint=0,title="Logistic Regression Weights")
                else:
                    pal = px.colors.qualitative.Set1
                    fig_c = go.Figure()
                    for i,cls in enumerate(classes):
                        cdf = pd.DataFrame({"Feature":fn[:coef.shape[1]],"Coef":coef[i]})
                        cdf = cdf.sort_values("Coef",key=abs,ascending=True)
                        fig_c.add_trace(go.Bar(x=cdf["Coef"],y=cdf["Feature"],
                                                orientation="h",name=f"class {cls}",
                                                marker_color=pal[i%len(pal)]))
                    fig_c.update_layout(barmode="group",title="Coefficients per Class")
                fig_c.update_layout(height=max(350, coef.shape[1]*22))
                st.plotly_chart(fig_c, use_container_width=True)

        with t_samp:
            n_show = st.slider("Rows to show", 10, min(100, len(y_te)), 20, 5,
                               key="n_show_clf")
            filter_opt = st.radio("Show", ["Random sample", "Misclassified only"],
                                  horizontal=True, key="filter_clf")
            samp_df = pd.DataFrame({
                "True":      y_te.astype(str),
                "Predicted": y_pred_te.astype(str),
                "Correct":   y_te == y_pred_te,
            })
            if y_prob_te is not None and S.task == "binary":
                samp_df["Prob(positive)"] = y_prob_te[:, 1].round(4)
            if filter_opt == "Misclassified only":
                samp_df = samp_df[~samp_df["Correct"]].head(n_show)
            else:
                samp_df = samp_df.sample(min(n_show, len(samp_df)), random_state=42)
            samp_df.index = range(1, len(samp_df) + 1)

            def _row_color(row):
                color = "" if row["Correct"] else "background-color: #ffe0e0"
                return [color] * len(row)

            st.dataframe(
                samp_df.style.apply(_row_color, axis=1),
                use_container_width=True,
            )
            n_wrong = int((~(y_te == y_pred_te)).sum())
            st.caption(f"{n_wrong} misclassified out of {len(y_te)} test samples "
                       f"({n_wrong/len(y_te)*100:.1f}% error rate)")

    st.button("Next →", on_click=advance, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Export / Deploy
# ══════════════════════════════════════════════════════════════════════════════
elif S.step == 5:
    st.title("Step 6 — Export & Deploy")
    if S.model is None:
        st.error("Train and evaluate a model first."); st.stop()

    # ── Bundle ────────────────────────────────────────────────────────────
    bundle = dict(preprocessor=S.preprocessor, model=S.model,
                  feature_cols=S.feature_cols, target_col=S.target_col,
                  task=S.task, best_params=S.best_params,
                  feature_names_out=S.feat_names_out)
    buf = io.BytesIO()
    pickle.dump(bundle, buf)
    buf.seek(0)

    # Predictions CSV
    pred_df = pd.DataFrame({"y_true": S.y_te, "y_pred": S.y_pred})
    if S.y_prob is not None and S.task == "binary":
        pred_df["prob_positive"] = S.y_prob[:,1]
    pred_csv = pred_df.to_csv(index=False).encode()

    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇  model.pkl", data=buf, file_name="model.pkl",
                        mime="application/octet-stream",
                        use_container_width=True, type="primary")
    c2.download_button("⬇  predictions.csv", data=pred_csv, file_name="predictions.csv",
                        mime="text/csv", use_container_width=True)

    # ── Pipeline summary ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Pipeline Summary")
    eff = [c for c in S.feature_cols
           if S.col_config_df is not None
           and c in S.col_config_df.set_index("Column").index
           and S.col_config_df.set_index("Column").loc[c,"Encoding"] != "Drop"]
    cfg_idx = S.col_config_df.set_index("Column")
    rows = []
    for c in eff:
        enc = cfg_idx.loc[c,"Encoding"]
        ord_ord = cfg_idx.loc[c,"Ordinal order (low → high)"] if enc == "Ordinal" else ""
        rows.append({"Column":c,"Encoding":enc,"Ordinal order":ord_ord})
    st.dataframe(pd.DataFrame(rows).set_index("Column"), use_container_width=True)

    if S.best_params:
        st.markdown("**Best hyperparameters:** " +
                    "  ·  ".join(f"`{k}={v}`" for k,v in S.best_params.items()))

    # ── Code snippet ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Deployment Code")
    is_torch_model = isinstance(S.model, (TorchMLPClassifier, TorchMLPRegressor))
    caption = "Copy this into any Python script to load the model and make predictions on new data."
    if is_torch_model:
        caption += " **Also copy `torch_mlp.py`** into the same folder — it defines the PyTorch model class needed to unpickle."
    st.caption(caption)

    has_proba = hasattr(S.model,"predict_proba") and S.task != "regression"
    torch_import = ("from torch_mlp import TorchMLPClassifier, TorchMLPRegressor  # required to unpickle\n"
                     if is_torch_model else "")
    snippet = textwrap.dedent(f"""
        import pickle
        import pandas as pd
        {torch_import}
        # ── Load the saved bundle ──────────────────────────────────────────
        with open("model.pkl", "rb") as f:
            bundle = pickle.load(f)

        preprocessor = bundle["preprocessor"]
        model        = bundle["model"]
        feature_cols = bundle["feature_cols"]
        task         = bundle["task"]   # '{S.task}'

        # ── Prepare new data ───────────────────────────────────────────────
        # Must be a DataFrame with these columns:
        #   {S.feature_cols}
        new_data = pd.read_csv("new_data.csv")

        X_new = preprocessor.transform(new_data[feature_cols])

        # ── Predict ───────────────────────────────────────────────────────
        predictions = model.predict(X_new)
        print(predictions)
        {"# Probability scores (binary / multiclass):" if has_proba else ""}
        {"probabilities = model.predict_proba(X_new)" if has_proba else ""}
    """).strip()

    st.code(snippet, language="python")

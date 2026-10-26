# -*- coding: utf-8 -*-
"""
Campus Facility Reservation — Data Mining Midterm (Streamlit version)

Run with:
    pip install streamlit pandas numpy matplotlib scikit-learn
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

st.set_page_config(page_title="Facility Reservation Data Mining", page_icon="📊", layout="wide")

BAD_STATUSES = ["cancelled", "no-show", "no show", "noshow"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def show_df(data, **kwargs):
    """Display a DataFrame safely (avoids Arrow errors with mixed-type columns)."""
    st.dataframe(data, use_container_width=True, **kwargs)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Part 0 — Synthetic dataset
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    facility_types = ["Gymnasium", "Auditorium", "Computer Laboratory", "Meeting Room", "Lecture Hall"]
    user_categories = ["Student", "Student Organization", "Faculty/Staff", "External Guest"]
    purposes = ["Meeting", "Practice", "Workshop", "Seminar", "Competition", "Event"]
    time_slots = ["Morning", "Afternoon", "Evening"]

    facility = rng.choice(facility_types, n, p=[0.22, 0.20, 0.23, 0.17, 0.18])
    user = rng.choice(user_categories, n, p=[0.38, 0.27, 0.25, 0.10])
    purpose = rng.choice(purposes, n)
    slot = rng.choice(time_slots, n, p=[0.32, 0.43, 0.25])
    lead = np.clip(rng.gamma(shape=2.2, scale=6.0, size=n).round().astype(int), 0, 60)
    participants = np.clip(rng.gamma(shape=2.3, scale=15, size=n).round().astype(int), 1, 180)

    reservation_dates = pd.to_datetime("2026-04-01") + pd.to_timedelta(rng.integers(0, 150, n), unit="D")
    submission_dates = reservation_dates - pd.to_timedelta(lead, unit="D")

    risk = (
        0.13
        + 0.18 * (lead <= 2)
        + 0.10 * (user == "External Guest")
        + 0.08 * (slot == "Evening")
        + 0.06 * (participants > 80)
    )
    risk = np.clip(risk, 0.08, 0.68)
    draw = rng.random(n)
    status = np.where(draw < risk * 0.58, "Cancelled",
                      np.where(draw < risk, "No-Show", "Used"))

    attendance, cancel_days = [], []
    for i, s in enumerate(status):
        if s == "Used":
            attendance.append(int(rng.binomial(int(participants[i]), 0.78)))
            cancel_days.append(np.nan)
        elif s == "Cancelled":
            attendance.append(0)
            cancel_days.append(int(rng.integers(0, max(1, min(int(lead[i]) + 1, 15)))))
        else:
            attendance.append(0)
            cancel_days.append(np.nan)

    df_original = pd.DataFrame({
        "Reservation_ID": [f"RES-{i:04d}" for i in range(1, n + 1)],
        "Facility_Type": facility,
        "User_Category": user,
        "Reservation_Date": reservation_dates.strftime("%Y-%m-%d"),
        "Submission_Date": submission_dates.strftime("%Y-%m-%d"),
        "Lead_Time_Days": lead,
        "Participants": participants,
        "Actual_Attendance": attendance,
        "Usage_Status": status,
        "Cancellation_Days_Before": cancel_days,
        "Purpose": purpose,
        "Time_Slot": slot,
    })

    # Small amount of realistic missingness for the cleaning tasks
    k_lead, k_purpose, k_status = (min(5, n), min(4, n), min(3, n))
    df_original.loc[rng.choice(n, size=k_lead, replace=False), "Lead_Time_Days"] = np.nan
    df_original.loc[rng.choice(n, size=k_purpose, replace=False), "Purpose"] = np.nan
    df_original.loc[rng.choice(n, size=k_status, replace=False), "Usage_Status"] = np.nan
    return df_original


# ----------------------------------------------------------------------------
# Part II — Cleaning and feature engineering
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_data(df_original: pd.DataFrame):
    df = df_original.copy(deep=True)

    for col in ["Facility_Type", "User_Category", "Purpose", "Time_Slot", "Usage_Status"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()

    status_map = {"used": "Used", "cancelled": "Cancelled", "canceled": "Cancelled",
                  "no-show": "No-Show", "no show": "No-Show", "noshow": "No-Show"}
    df["Usage_Status"] = df["Usage_Status"].map(
        lambda x: status_map.get(str(x).strip().lower(), x) if pd.notna(x) else pd.NA
    )

    for col in ["Facility_Type", "User_Category", "Purpose", "Time_Slot"]:
        df[col] = df[col].map(lambda x: str(x).title().strip() if pd.notna(x) else pd.NA).astype("string")

    rows_before = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for col in ["Reservation_Date", "Submission_Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Lead_Time_Days", "Participants", "Actual_Attendance", "Cancellation_Days_Before"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["Lead_Time_Days"] < 0, "Lead_Time_Days"] = np.nan
    df.loc[df["Participants"] <= 0, "Participants"] = np.nan
    invalid_attendance = (
        df["Actual_Attendance"].lt(0)
        | (df["Actual_Attendance"].notna() & df["Participants"].notna()
           & df["Actual_Attendance"].gt(df["Participants"]))
    )
    df.loc[invalid_attendance, "Actual_Attendance"] = np.nan
    df.loc[df["Cancellation_Days_Before"] < 0, "Cancellation_Days_Before"] = np.nan

    date_lead = (df["Reservation_Date"] - df["Submission_Date"]).dt.days
    mismatch = date_lead.notna() & df["Lead_Time_Days"].notna() & date_lead.ne(df["Lead_Time_Days"])
    mismatch_count = int(mismatch.sum())

    # Task 5 — derived variables
    df["Lead_Time_Category"] = pd.cut(
        df["Lead_Time_Days"], bins=[-np.inf, 2, 7, 14, np.inf],
        labels=["Last Minute", "Short Lead", "Medium Lead", "Early Booking"],
    )
    df["Reservation_Outcome"] = df["Usage_Status"].map(
        {"Used": "Completed", "Cancelled": "Not Completed", "No-Show": "Not Completed"}
    )
    denominator = df["Participants"].where(df["Participants"] > 0)
    df["Attendance_Ratio"] = df["Actual_Attendance"].div(denominator)
    df.loc[~np.isfinite(df["Attendance_Ratio"]), "Attendance_Ratio"] = np.nan

    info = {
        "rows_before": rows_before,
        "duplicate_rows": duplicate_rows,
        "rows_after": len(df),
        "mismatch_count": mismatch_count,
    }
    return df, info


# ----------------------------------------------------------------------------
# Part IV — Decision tree model
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def train_model(df: pd.DataFrame, seed: int, max_depth: int, min_leaf: int):
    model_df = df.copy()
    for date_col, prefix in [("Reservation_Date", "Reservation"), ("Submission_Date", "Submission")]:
        parsed = pd.to_datetime(model_df[date_col], errors="coerce")
        model_df[f"{prefix}_Month"] = parsed.dt.month
        model_df[f"{prefix}_DayOfWeek"] = parsed.dt.dayofweek

    predictors = [c for c in [
        "Facility_Type", "User_Category", "Lead_Time_Days", "Participants",
        "Purpose", "Time_Slot", "Reservation_Month", "Reservation_DayOfWeek",
        "Submission_Month", "Submission_DayOfWeek",
    ] if c in model_df.columns]

    model_df = model_df[model_df["Reservation_Outcome"].notna()].copy()
    X = model_df[predictors].copy()
    for col in predictors:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        else:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    y = model_df["Reservation_Outcome"].astype(str).copy()

    if len(predictors) < 5:
        raise ValueError("At least five submission-time predictors are required.")
    if y.nunique() < 2:
        raise ValueError("Target contains fewer than two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    categorical_features = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_features = [c for c in X_train.columns if c not in categorical_features]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])
    tree = DecisionTreeClassifier(random_state=seed, class_weight="balanced",
                                  max_depth=max_depth, min_samples_leaf=min_leaf)
    clf = Pipeline([("preprocessor", preprocessor), ("model", tree)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Aggregate one-hot importances back to original predictors
    encoded_names = clf.named_steps["preprocessor"].get_feature_names_out()
    importances = clf.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"Encoded feature": encoded_names, "Importance": importances})

    def original_predictor(encoded_name):
        name = encoded_name.split("__", 1)[-1]
        for feature in sorted(predictors, key=len, reverse=True):
            if name == feature or name.startswith(feature + "_"):
                return feature
        return name

    importance_df["Original predictor"] = importance_df["Encoded feature"].apply(original_predictor)
    grouped_importance = (importance_df.groupby("Original predictor", as_index=False)["Importance"]
                          .sum().sort_values("Importance", ascending=False).reset_index(drop=True))

    return {
        "predictors": predictors,
        "y": y,
        "X_train_len": len(X_train),
        "X_test_len": len(X_test),
        "y_test": y_test,
        "y_pred": y_pred,
        "grouped_importance": grouped_importance,
    }


# ----------------------------------------------------------------------------
# EDA helpers
# ----------------------------------------------------------------------------
def user_category_rates(df: pd.DataFrame) -> pd.DataFrame:
    status_lower = df["Usage_Status"].astype("string").str.lower()
    tmp = df.assign(Is_Cancelled_NoShow=status_lower.isin(BAD_STATUSES))
    valid = tmp[tmp["Usage_Status"].notna()]
    rates = valid.groupby("User_Category", dropna=False).agg(
        Total_with_known_status=("Is_Cancelled_NoShow", "size"),
        Cancelled_or_NoShow=("Is_Cancelled_NoShow", "sum"),
    )
    rates["Cancellation_NoShow_Percent"] = 100 * rates["Cancelled_or_NoShow"] / rates["Total_with_known_status"]
    return rates.sort_values("Cancellation_NoShow_Percent", ascending=False)


# ----------------------------------------------------------------------------
# UML diagram
# ----------------------------------------------------------------------------
def draw_uml_figure():
    use_cases = {
        "uc1":  ("Submit Reservation\nRequest",           330, 105, 210, 70, "#dbeafe", "#2563eb"),
        "uc2":  ("Cancel Reservation",                    330, 205, 210, 70, "#dbeafe", "#2563eb"),
        "uc3":  ("View Reservation\nStatus",              330, 305, 210, 70, "#dbeafe", "#2563eb"),
        "uc9":  ("Validate & Clean\nReservation Data",    330, 575, 210, 70, "#fef3c7", "#d97706"),
        "uc10": ("Train & Evaluate\nDecision Tree Model", 330, 695, 210, 70, "#fef3c7", "#d97706"),
        "uc4":  ("View Reservation Statistics\nby Facility Type", 730, 105, 230, 70, "#dcfce7", "#16a34a"),
        "uc5":  ("Analyze Cancellation /\nNo-Show Patterns",      730, 205, 230, 70, "#dcfce7", "#16a34a"),
        "uc8":  ("Review & Decide on\nReservation (Manual)",      730, 325, 230, 70, "#dcfce7", "#16a34a"),
        "uc6":  ("View Predicted\nNon-Completion Risk",           730, 455, 230, 70, "#fce7f3", "#db2777"),
        "uc7":  ("Review Prediction\nExplanation",                730, 575, 230, 70, "#fce7f3", "#db2777"),
    }
    actors = {
        "req":     ("Reservation Requester\n(Student / Org /\nFaculty / Guest)", 75, 150),
        "admin":   ("Facility\nAdministrator", 965, 290),
        "analyst": ("Data Analyst", 75, 590),
    }
    assoc = [("req", "uc1"), ("req", "uc2"), ("req", "uc3"),
             ("admin", "uc4"), ("admin", "uc5"), ("admin", "uc8"), ("admin", "uc6"), ("admin", "uc7"),
             ("analyst", "uc9"), ("analyst", "uc10")]
    relations = [("uc8", "uc6", "«include»"), ("uc7", "uc6", "«extend»"), ("uc10", "uc9", "«include»")]

    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.set_xlim(0, 1040); ax.set_ylim(780, 0); ax.axis("off")
    ax.add_patch(Rectangle((190, 20), 680, 740, fill=True, fc="#f8fafc", ec="#334155", lw=2, zorder=0))
    ax.text(530, 42, "Campus Facility Reservation and Monitoring System",
            ha="center", va="center", fontsize=12, fontweight="bold")

    def actor_anchor(a):
        _, x, y = actors[a]
        return np.array([x, y + 28.0])

    def edge_point(center, a, b, toward):
        d = np.array(toward) - np.array(center)
        k = 1 / np.sqrt((d[0] / a) ** 2 + (d[1] / b) ** 2)
        return np.array(center) + d * k

    for a_id, u_id in assoc:
        _, cx, cy, w, h, _, _ = use_cases[u_id]
        p0 = actor_anchor(a_id)
        p1 = edge_point((cx, cy), w / 2, h / 2, p0)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#111827", lw=1.2, zorder=1)

    for src, dst, label in relations:
        _, sx, sy, sw, sh, _, _ = use_cases[src]
        _, dx, dy, dw, dh, _, _ = use_cases[dst]
        p0 = edge_point((sx, sy), sw / 2, sh / 2, (dx, dy))
        p1 = edge_point((dx, dy), dw / 2, dh / 2, (sx, sy))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=14,
                                     linestyle="--", color="#111827", lw=1.2, zorder=3))
        mid = (p0 + p1) / 2
        ax.text(mid[0] + 8, mid[1], label, fontsize=9, style="italic", va="center",
                bbox=dict(fc="#f8fafc", ec="none", pad=1), zorder=4)

    for _, (label, cx, cy, w, h, fc, ec) in use_cases.items():
        ax.add_patch(Ellipse((cx, cy), w, h, fc=fc, ec=ec, lw=1.5, zorder=2))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9, zorder=5)

    for _, (label, x, y) in actors.items():
        ax.add_patch(Ellipse((x, y + 8), 16, 16, fc="white", ec="black", lw=1.5, zorder=2))
        ax.plot([x, x], [y + 16, y + 38], "k", lw=1.5)
        ax.plot([x - 14, x + 14], [y + 24, y + 24], "k", lw=1.5)
        ax.plot([x, x - 12], [y + 38, y + 58], "k", lw=1.5)
        ax.plot([x, x + 12], [y + 38, y + 58], "k", lw=1.5)
        ax.text(x, y + 66, label, ha="center", va="top", fontsize=9)

    ax.add_patch(Rectangle((615, 650), 230, 80, fc="#fff9c4", ec="#a16207", lw=1, zorder=2))
    ax.text(623, 690,
            "Note: Risk prediction is advisory only.\nThe system never automatically rejects a\n"
            "reservation; an administrator makes the\nfinal decision.",
            fontsize=8, va="center", zorder=5)
    ax.set_title("Figure — UML Use Case Diagram: Campus Facility Reservation and Monitoring System",
                 fontsize=11, pad=8)
    fig.tight_layout()
    return fig


DRAWIO_XML = r"""<mxfile host="app.diagrams.net">
  <diagram id="uc1" name="Use Case Diagram">
    <mxGraphModel dx="1000" dy="780" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="800" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>

        <!-- System boundary -->
        <mxCell id="boundary" value="Campus Facility Reservation and Monitoring System" style="rounded=0;whiteSpace=wrap;html=1;verticalAlign=top;fontStyle=1;fontSize=14;fillColor=#f8fafc;strokeColor=#334155;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="190" y="20" width="680" height="740" as="geometry"/>
        </mxCell>

        <!-- Actors -->
        <mxCell id="actorReq" value="Reservation Requester&lt;br&gt;(Student / Org / Faculty / Guest)" style="shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;" vertex="1" parent="1">
          <mxGeometry x="60" y="150" width="30" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="actorAdmin" value="Facility Administrator" style="shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;" vertex="1" parent="1">
          <mxGeometry x="950" y="290" width="30" height="60" as="geometry"/>
        </mxCell>
        <mxCell id="actorAnalyst" value="Data Analyst" style="shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;" vertex="1" parent="1">
          <mxGeometry x="60" y="590" width="30" height="60" as="geometry"/>
        </mxCell>

        <!-- Left column use cases (requester / analyst) -->
        <mxCell id="uc1" value="Submit Reservation Request" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;" vertex="1" parent="1">
          <mxGeometry x="225" y="70" width="210" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="uc2" value="Cancel Reservation" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;" vertex="1" parent="1">
          <mxGeometry x="225" y="170" width="210" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="uc3" value="View Reservation Status" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dbeafe;strokeColor=#2563eb;" vertex="1" parent="1">
          <mxGeometry x="225" y="270" width="210" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="uc9" value="Validate &amp;amp; Clean Reservation Data" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fef3c7;strokeColor=#d97706;" vertex="1" parent="1">
          <mxGeometry x="225" y="540" width="210" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="uc10" value="Train &amp;amp; Evaluate Decision Tree Model" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fef3c7;strokeColor=#d97706;" vertex="1" parent="1">
          <mxGeometry x="225" y="660" width="210" height="70" as="geometry"/>
        </mxCell>

        <!-- Right column use cases (administrator / analytics) -->
        <mxCell id="uc4" value="View Reservation Statistics by Facility Type" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dcfce7;strokeColor=#16a34a;" vertex="1" parent="1">
          <mxGeometry x="615" y="70" width="230" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="uc5" value="Analyze Cancellation / No-Show Patterns" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dcfce7;strokeColor=#16a34a;" vertex="1" parent="1">
          <mxGeometry x="615" y="170" width="230" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="uc8" value="Review &amp;amp; Decide on Reservation (Manual)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dcfce7;strokeColor=#16a34a;" vertex="1" parent="1">
          <mxGeometry x="615" y="290" width="230" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="uc6" value="View Predicted Non-Completion Risk" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fce7f3;strokeColor=#db2777;" vertex="1" parent="1">
          <mxGeometry x="615" y="420" width="230" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="uc7" value="Review Prediction Explanation" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fce7f3;strokeColor=#db2777;" vertex="1" parent="1">
          <mxGeometry x="615" y="540" width="230" height="70" as="geometry"/>
        </mxCell>

        <!-- Advisory note -->
        <mxCell id="note" value="Note: Risk prediction is advisory only. The system never automatically rejects a reservation; an administrator makes the final decision." style="shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff9c4;strokeColor=#a16207;fontSize=11;align=left;spacingLeft=6;" vertex="1" parent="1">
          <mxGeometry x="615" y="650" width="230" height="80" as="geometry"/>
        </mxCell>

        <!-- Associations: Requester -->
        <mxCell id="a1" style="endArrow=none;html=1;" edge="1" parent="1" source="actorReq" target="uc1"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="a2" style="endArrow=none;html=1;" edge="1" parent="1" source="actorReq" target="uc2"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="a3" style="endArrow=none;html=1;" edge="1" parent="1" source="actorReq" target="uc3"><mxGeometry relative="1" as="geometry"/></mxCell>

        <!-- Associations: Administrator -->
        <mxCell id="a4" style="endArrow=none;html=1;" edge="1" parent="1" source="actorAdmin" target="uc4"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="a5" style="endArrow=none;html=1;" edge="1" parent="1" source="actorAdmin" target="uc5"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="a6" style="endArrow=none;html=1;" edge="1" parent="1" source="actorAdmin" target="uc8"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="a7" style="endArrow=none;html=1;" edge="1" parent="1" source="actorAdmin" target="uc6"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="a8" style="endArrow=none;html=1;" edge="1" parent="1" source="actorAdmin" target="uc7"><mxGeometry relative="1" as="geometry"/></mxCell>

        <!-- Associations: Data Analyst -->
        <mxCell id="a9" style="endArrow=none;html=1;" edge="1" parent="1" source="actorAnalyst" target="uc9"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="a10" style="endArrow=none;html=1;" edge="1" parent="1" source="actorAnalyst" target="uc10"><mxGeometry relative="1" as="geometry"/></mxCell>

        <!-- Include / extend relationships -->
        <mxCell id="i1" value="«include»" style="endArrow=open;dashed=1;html=1;endFill=0;labelBackgroundColor=#f8fafc;fontSize=11;" edge="1" parent="1" source="uc8" target="uc6"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="e1" value="«extend»" style="endArrow=open;dashed=1;html=1;endFill=0;labelBackgroundColor=#f8fafc;fontSize=11;" edge="1" parent="1" source="uc7" target="uc6"><mxGeometry relative="1" as="geometry"/></mxCell>
        <mxCell id="i2" value="«include»" style="endArrow=open;dashed=1;html=1;endFill=0;labelBackgroundColor=#f8fafc;fontSize=11;" edge="1" parent="1" source="uc10" target="uc9"><mxGeometry relative="1" as="geometry"/></mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
RANDOM_SEED = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
N_ROWS = st.sidebar.number_input("Number of records", min_value=50, max_value=5000, value=402, step=1)
st.sidebar.markdown("**Decision tree**")
MAX_DEPTH = st.sidebar.slider("max_depth", 2, 12, 5)
MIN_LEAF = st.sidebar.slider("min_samples_leaf", 1, 30, 5)
st.sidebar.caption("Defaults (seed 42, 402 rows, depth 5, leaf 5) reproduce the original notebook.")

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
df_original = generate_data(int(N_ROWS), int(RANDOM_SEED))
df, clean_info = clean_data(df_original)
data_source = f"Reproducibly generated synthetic sample ({int(N_ROWS)} rows; seed={int(RANDOM_SEED)})"

st.title("📊 Campus Facility Reservation — Data Mining Midterm")
st.caption(f"Dataset ready: {df_original.shape[0]} rows × {df_original.shape[1]} columns  |  Source: {data_source}")

tabs = st.tabs([
    "I · Data Understanding",
    "II · Cleaning & Features",
    "III · EDA",
    "IV · Modeling",
    "V · Requirements & UML",
    "VI · Decision Support",
    "Export",
])

# ============================================================================
# Part I
# ============================================================================
with tabs[0]:
    st.header("Part I — Dataset Acquisition and Understanding")

    st.subheader("Task 1 — Preview, dimensions, types, and summary statistics")
    st.markdown("**First 10 records**")
    show_df(df_original.head(10))
    c1, c2 = st.columns(2)
    c1.metric("Rows", df_original.shape[0])
    c2.metric("Columns", df_original.shape[1])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Column data types**")
        show_df(df_original.dtypes.astype(str).rename("Data type").to_frame())
    with col_b:
        st.markdown("**Summary statistics**")
        show_df(df_original.describe(include="all").T.astype(str))

    st.subheader("Task 2 — Predictors, target, categorical and numerical variables")
    st.info(
        "`Reservation_ID` is an identifier, not a useful behavioral predictor. `Usage_Status` is used to "
        "construct the target. `Actual_Attendance` and `Cancellation_Days_Before` are post-outcome fields "
        "and must not be predictors."
    )
    candidate_predictors = [
        c for c in ["Facility_Type", "User_Category", "Lead_Time_Days", "Participants",
                    "Purpose", "Time_Slot", "Reservation_Date", "Submission_Date"]
        if c in df_original.columns
    ]
    categorical_vars = df_original.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical_vars = df_original.select_dtypes(include=np.number).columns.tolist()

    st.markdown("**Predictor candidates (verify available at submission time):**")
    st.markdown("\n".join(f"- {c}" for c in candidate_predictors))
    st.markdown("**Target source:** `Usage_Status` → `Reservation_Outcome`")
    st.markdown(f"**Categorical variables:** {categorical_vars}")
    st.markdown(f"**Numerical variables:** {numerical_vars}")

    st.subheader("Task 3 — Data acquisition and validation")
    st.markdown(
        "A production system could retrieve authorized records from an authenticated API or parameterized "
        "SQL query. Use access controls and request only necessary fields. Validate required columns, date "
        "formats, permitted status values, and identifier uniqueness before analysis."
    )
    required_columns = ["Reservation_ID", "Facility_Type", "User_Category", "Reservation_Date",
                        "Submission_Date", "Lead_Time_Days", "Participants", "Actual_Attendance",
                        "Usage_Status", "Cancellation_Days_Before", "Purpose", "Time_Slot"]
    missing_required = [c for c in required_columns if c not in df_original.columns]
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Required columns missing", len(missing_required))
    v2.metric("Duplicate Reservation_ID", int(df_original["Reservation_ID"].duplicated().sum()))
    v3.metric("Unparseable Reservation_Date",
              int(pd.to_datetime(df_original["Reservation_Date"], errors="coerce").isna().sum()))
    v4.metric("Unparseable Submission_Date",
              int(pd.to_datetime(df_original["Submission_Date"], errors="coerce").isna().sum()))

# ============================================================================
# Part II
# ============================================================================
with tabs[1]:
    st.header("Part II — Data Cleaning and Feature Engineering")

    st.subheader("Task 4 — Data quality checks and cleaning decisions")
    st.markdown(
        "The original data is preserved in `df_original`. Cleaning is conservative: standardize labels, "
        "remove exact duplicate rows, parse dates and numerics, and mark invalid numeric values as missing "
        "rather than inventing replacements. Missing cancellation lead time is retained as missing because "
        "it may indicate no cancellation occurred."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows before cleaning", clean_info["rows_before"])
    m2.metric("Duplicate rows removed", clean_info["duplicate_rows"])
    m3.metric("Rows after cleaning", clean_info["rows_after"])
    m4.metric("Lead-time/date mismatches", clean_info["mismatch_count"])
    st.caption("Mismatches are retained for audit.")

    st.markdown("**Missing values after cleaning**")
    show_df(df.isna().sum().rename("Missing count").to_frame())

    st.subheader("Task 5 — Derived variables")
    st.markdown(
        "- `Lead_Time_Category`: 0–2 = Last Minute; 3–7 = Short Lead; 8–14 = Medium Lead; >14 = Early Booking.\n"
        "- `Reservation_Outcome`: Completed for Used; Not Completed for Cancelled or No-Show; other statuses remain missing.\n"
        "- `Attendance_Ratio`: Actual_Attendance / Participants; missing if numerator/denominator is missing "
        "or denominator is not positive."
    )
    preview_cols = [c for c in ["Lead_Time_Days", "Lead_Time_Category", "Usage_Status",
                                "Reservation_Outcome", "Participants", "Actual_Attendance",
                                "Attendance_Ratio"] if c in df]
    show_df(df[preview_cols].head(10).astype({"Lead_Time_Category": "string"}))

# ============================================================================
# Part III
# ============================================================================
with tabs[2]:
    st.header("Part III — Exploratory Data Analysis")
    st.caption(
        "Percentages use the number of reservations in the relevant group with known `Usage_Status` as the "
        "denominator. These charts describe historical association, not causation."
    )

    # Facility counts
    st.subheader("Reservation count by facility type")
    facility_counts = df["Facility_Type"].value_counts(dropna=False)
    c1, c2 = st.columns([1, 2])
    with c1:
        show_df(facility_counts.rename("Reservation count").to_frame())
        st.success(f"Highest-count facility type: **{facility_counts.index[0]}** — "
                   f"{int(facility_counts.iloc[0])} reservations")
    with c2:
        fig, ax = plt.subplots(figsize=(9, 4))
        facility_counts.sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_title("Reservation Count by Facility Type")
        ax.set_xlabel("Facility Type"); ax.set_ylabel("Number of Reservations")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        show_fig(fig)

    # User category rates
    st.subheader("Cancellation/no-show rate by user category")
    user_rates = user_category_rates(df)
    c1, c2 = st.columns([1, 2])
    with c1:
        show_df(user_rates.round(2))
        if len(user_rates):
            top = user_rates.iloc[0]
            st.success(
                f"Highest cancellation/no-show rate: **{user_rates.index[0]}** — "
                f"{int(top['Cancelled_or_NoShow'])}/{int(top['Total_with_known_status'])} known-status "
                f"reservations ({top['Cancellation_NoShow_Percent']:.1f}%)."
            )
    with c2:
        fig, ax = plt.subplots(figsize=(9, 4))
        user_rates["Cancellation_NoShow_Percent"].plot(kind="bar", ax=ax)
        ax.set_title("Cancellation/No-Show Rate by User Category")
        ax.set_xlabel("User Category"); ax.set_ylabel("Cancelled or No-Show (%)"); ax.set_ylim(0, 100)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        show_fig(fig)

    # Lead-time vs outcome
    st.subheader("Reservation outcome by lead-time category")
    cross = pd.crosstab(df["Lead_Time_Category"], df["Reservation_Outcome"])
    row_pct = pd.crosstab(df["Lead_Time_Category"], df["Reservation_Outcome"], normalize="index") * 100
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Counts**")
        show_df(cross)
        st.markdown("**Row percentages**")
        show_df(row_pct.round(1).add_suffix(" (%)"))
    with c2:
        fig, ax = plt.subplots(figsize=(9, 4))
        row_pct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Reservation Outcome by Lead-Time Category")
        ax.set_xlabel("Lead-Time Category"); ax.set_ylabel("Percent within Lead-Time Category")
        ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        show_fig(fig)

    st.info(
        "**Interpretation:** Record the highest facility count, the highest user-category "
        "cancellation/no-show rate with numerator/denominator, and the lead-time category with the highest "
        "`Not Completed` share. Use the displayed values, not assumptions, and do not claim causation."
    )

# ============================================================================
# Part IV
# ============================================================================
with tabs[3]:
    st.header("Part IV — Data Mining and Predictive Modeling")
    st.subheader("Tasks 7–8 — Decision Tree classifier")
    st.markdown(
        "**Leakage prevention:** `Usage_Status` creates the target but is not a predictor. "
        "`Actual_Attendance`, `Cancellation_Days_Before`, `Attendance_Ratio`, and outcome-derived fields are "
        "excluded. Date fields are represented by month and day of week."
    )

    try:
        res = train_model(df, int(RANDOM_SEED), int(MAX_DEPTH), int(MIN_LEAF))
        model_ok = True
    except ValueError as e:
        st.error(str(e))
        model_ok = False

    if model_ok:
        predictors = res["predictors"]
        y_test, y_pred = res["y_test"], res["y_pred"]
        grouped_importance = res["grouped_importance"]
        labels = ["Completed", "Not Completed"]

        st.markdown(f"**Predictors ({len(predictors)}):** {predictors}")
        st.caption("Excluded from predictors: Usage_Status, Actual_Attendance, Cancellation_Days_Before, "
                   "Attendance_Ratio, Reservation_Outcome")

        c1, c2, c3 = st.columns(3)
        c1.markdown("**Target class distribution**")
        c1.dataframe(res["y"].value_counts().rename("Count").to_frame(), use_container_width=True)
        c2.metric("Training rows", res["X_train_len"])
        c3.metric("Test rows", res["X_test_len"])

        st.subheader("Evaluation")
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        a2.metric("Precision (macro)", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        a3.metric("Recall (macro)", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
        a4.metric("F1 (macro)", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix**")
            show_df(pd.DataFrame(cm, index=[f"Actual {x}" for x in labels],
                                 columns=[f"Predicted {x}" for x in labels]))
            st.markdown("**Class-specific report**")
            report = classification_report(y_test, y_pred, labels=labels, zero_division=0, output_dict=True)
            show_df(pd.DataFrame(report).T.round(3))
        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, values_format="d")
            ax.set_title("Decision Tree — Confusion Matrix")
            fig.tight_layout()
            show_fig(fig)

        st.subheader("Feature importance")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Three most influential original predictors**")
            show_df(grouped_importance.head(3))
        with c2:
            fig, ax = plt.subplots(figsize=(8, 4))
            grouped_importance.head(10).sort_values("Importance").plot(
                x="Original predictor", y="Importance", kind="barh", legend=False, ax=ax)
            ax.set_title("Feature Importance by Original Predictor")
            ax.set_xlabel("Aggregated impurity-based importance"); ax.set_ylabel("Predictor")
            fig.tight_layout()
            show_fig(fig)

        st.subheader("Task 8 — Reliability assessment")
        st.markdown(
            "Do not judge reliability by accuracy alone. Inspect the confusion matrix and class-specific "
            "recall/F1 for `Not Completed`. The dataset is synthetic and may be imbalanced; one train/test "
            "split may not generalize. Improvements include stratified cross-validation, comparison with a "
            "simple baseline and alternative models, and testing on newer validated data. Predictions should "
            "support human review, not automatic rejection."
        )

# ============================================================================
# Part V
# ============================================================================
with tabs[4]:
    st.header("Part V — Requirements Analysis and Use Case Modeling")

    st.subheader("Task 9 — Functional requirements tied to actual findings")
    st.caption("Evidence statements are generated from this run's descriptive analysis. "
               "Make sure each is supported by the counts and rates shown in the EDA tab.")

    evidence = []
    if df["Facility_Type"].notna().any():
        counts = df["Facility_Type"].value_counts()
        evidence.append(("Reservation volume",
                         f"{counts.index[0]} has the highest observed count ({int(counts.iloc[0])} reservations)."))
    rates = user_category_rates(df)
    if len(rates):
        r = rates.iloc[0]
        evidence.append(("Cancellation/no-show pattern",
                         f"{rates.index[0]}: {int(r['Cancelled_or_NoShow'])}/{int(r['Total_with_known_status'])} "
                         f"known-status reservations ({r['Cancellation_NoShow_Percent']:.1f}%)."))
    tab = pd.crosstab(df["Lead_Time_Category"], df["Reservation_Outcome"], normalize="index") * 100
    if "Not Completed" in tab and len(tab):
        cat = tab["Not Completed"].idxmax()
        evidence.append(("Lead-time/outcome pattern",
                         f"{cat} has the highest observed Not Completed share "
                         f"({tab.loc[cat, 'Not Completed']:.1f}% within that category)."))

    requirements = [
        ("FR-DM-01", "The system shall display reservation counts by facility type to help administrators monitor demand."),
        ("FR-DM-02", "The system shall display cancellation/no-show rates by user category with numerator and denominator."),
        ("FR-DM-03", "The system shall show estimated non-completion risk, the main contributing predictors, and a human-review notice."),
    ]
    requirements_df = pd.DataFrame([
        {"Finding / Evidence": evidence[i][1] if i < len(evidence)
         else "Confirm a supporting finding from current EDA/model output before submission.",
         "Requirement ID": rid, "Functional Requirement": req}
        for i, (rid, req) in enumerate(requirements)
    ])
    show_df(requirements_df)

    st.subheader("Task 10 — UML use case diagram")
    st.markdown(
        "Shows the system boundary, three actors, ten use cases, associations, and include/extend "
        "relationships. Risk prediction is advisory; the diagram does not automate rejection. "
        "The editable `.drawio` file can be opened in [diagrams.net](https://app.diagrams.net/)."
    )
    uml_fig = draw_uml_figure()
    show_fig_buf = io.BytesIO()
    uml_fig.savefig(show_fig_buf, format="png", dpi=200, bbox_inches="tight")
    st.pyplot(uml_fig)
    plt.close(uml_fig)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download PNG", show_fig_buf.getvalue(),
                       file_name="Campus_Facility_Reservation_UML.png", mime="image/png")
    d2.download_button("⬇️ Download .drawio", DRAWIO_XML.encode("utf-8"),
                       file_name="Campus_Facility_Reservation_UML.drawio", mime="application/xml")

# ============================================================================
# Part VI
# ============================================================================
with tabs[5]:
    st.header("Part VI — Decision Support and Critical Analysis")

    st.subheader("Task 11 — Dashboard wireframe")
    st.caption("Separates **observed historical measures** from **model predictions**. Predicted risk should be "
               "labeled as an estimate, with model version/date and explanation.")

    total_reservations = len(df)
    known_status = int(df["Usage_Status"].notna().sum())
    status_lower = df["Usage_Status"].astype("string").str.lower()
    bad_count = int(status_lower.isin(BAD_STATUSES).sum())
    bad_rate = 100 * bad_count / known_status if known_status else np.nan

    with st.container(border=True):
        st.markdown("#### CAMPUS FACILITY RESERVATION & MONITORING — ADMIN DASHBOARD WIREFRAME")
        st.markdown("**OBSERVED FACTS**")
        k1, k2, k3 = st.columns(3)
        k1.metric("Total reservations", total_reservations)
        k2.metric("Known outcome status", known_status)
        if known_status:
            k3.metric("Cancelled / no-show", f"{bad_count} ({bad_rate:.1f}%)")
        st.markdown("**OBSERVED TRENDS:** facility counts; cancellation/no-show by user category; "
                    "outcomes by lead-time category.")
        st.divider()
        st.warning(
            "**PREDICTED RISK — ESTIMATE, NOT OBSERVED FACT**\n\n"
            "Show probability/risk band, model version and timestamp, contributing factors, and administrator "
            "review. Do not automatically reject a reservation based only on a model score."
        )

    st.subheader("Task 12 — Short answers")
    st.markdown("**(a) Why should predicted high risk not automatically reject a reservation?**")
    st.write("A prediction is uncertain and can be wrong. Automatic rejection may unfairly deny legitimate "
             "access. Use risk as a review aid, explain the factors, and let an authorized administrator "
             "apply transparent policy.")
    st.markdown("**(b) How does poor data quality affect the model and information system?**")
    st.write("Missing, duplicated, inconsistent, or invalid records can distort counts and rates, bias "
             "training, reduce predictive quality, and lead administrators to make decisions from unreliable "
             "information. Validation and documented cleaning are essential.")
    st.markdown("**(c) What information should explain a high-risk prediction?**")
    st.write("Display the estimated probability and risk band, model version/date, contributing factors, "
             "relevant historical context, uncertainty and limitations, and a clear statement that the "
             "prediction is advisory and requires human review.")

# ============================================================================
# Export
# ============================================================================
with tabs[6]:
    st.header("Export cleaned data and data-quality summary")

    quality_summary = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Missing_Percent": (100 * df.isna().mean()).round(2),
        "Data_Type": df.dtypes.astype(str),
    })
    show_df(quality_summary)

    report_lines = [
        "DATA MINING LABORATORY — RUN SUMMARY",
        f"Source: {data_source}",
        f"Original shape: {df_original.shape}",
        f"Cleaned shape: {df.shape}",
        f"Exact duplicates removed: {clean_info['duplicate_rows']}",
        "",
        "Data-quality summary:",
        quality_summary.to_string(),
        "",
        "Reservation counts by facility type:",
        df["Facility_Type"].value_counts(dropna=False).to_string(),
    ]
    try:
        res = train_model(df, int(RANDOM_SEED), int(MAX_DEPTH), int(MIN_LEAF))
        yt, yp = res["y_test"], res["y_pred"]
        report_lines += [
            "", "Top model predictors:", res["grouped_importance"].head(3).to_string(index=False),
            "", "Model evaluation:",
            f"Accuracy: {accuracy_score(yt, yp):.4f}",
            f"Macro precision: {precision_score(yt, yp, average='macro', zero_division=0):.4f}",
            f"Macro recall: {recall_score(yt, yp, average='macro', zero_division=0):.4f}",
            f"Macro F1: {f1_score(yt, yp, average='macro', zero_division=0):.4f}",
            "Inspect class-specific results; accuracy alone is insufficient.",
        ]
    except ValueError:
        pass
    report_text = "\n".join(report_lines)

    e1, e2, e3 = st.columns(3)
    e1.download_button("⬇️ facility_reservations_cleaned.csv", df.to_csv(index=False).encode("utf-8"),
                       file_name="facility_reservations_cleaned.csv", mime="text/csv")
    e2.download_button("⬇️ data_quality_summary.csv", quality_summary.to_csv(index=True).encode("utf-8"),
                       file_name="data_quality_summary.csv", mime="text/csv")
    e3.download_button("⬇️ findings_report.txt", report_text.encode("utf-8"),
                       file_name="findings_report.txt", mime="text/plain")

    with st.expander("Preview findings report"):
        st.code(report_text)

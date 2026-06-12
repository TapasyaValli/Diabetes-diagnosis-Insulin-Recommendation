import json
import math
import os
import pickle
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request as urlrequest
from urllib.error import URLError
from xml.etree import ElementTree as ET

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn


ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_DIABETES_CSV = Path(
    os.getenv(
        "DIABETES_DATASET",
        str(ROOT / "data" / "diabetes_realistic_dataset.csv"),
    )
)
DEFAULT_CGM_ZIP = Path(
    os.getenv(
        "CGM_ARCHIVE",
        str(ROOT / "data" / "cgm_archive.zip"),
    )
)
PRETRAINED_RF = Path(
    os.getenv(
        "DIABETES_RF_MODEL",
        r"C:\Users\DELL\Downloads\datasets_inhouse\diabetes_rf_model.pkl",
    )
)
PRETRAINED_FEATURES = Path(
    os.getenv(
        "SELECTED_FEATURES",
        str(ROOT / "data" / "selected_features.pkl"),
    )
)

CLASS_NAMES = {
    0: "Non-diabetic",
    1: "Type 1 diabetes",
    2: "Type 2 diabetes",
}
DIABETIC_CLASSES = {"Type 1 diabetes", "Type 2 diabetes"}


class DoseLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 24):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Linear(24, 3),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


@dataclass
class LSTMArtifacts:
    model: DoseLSTM
    scaler_mean: float
    scaler_std: float
    patient_table: pd.DataFrame
    training_summary: Dict[str, float]


def patient_id_for_index(index: int) -> str:
    return f"PT-{index + 1:06d}"


@st.cache_data(show_spinner=False)
def load_diabetes_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df.insert(0, "Patient_ID", [patient_id_for_index(i) for i in range(len(df))])
    return df


def load_selected_features(df: pd.DataFrame) -> List[str]:
    if PRETRAINED_FEATURES.exists():
        with open(PRETRAINED_FEATURES, "rb") as f:
            features = pickle.load(f)
        return [feature for feature in features if feature in df.columns]

    candidates = [c for c in df.columns if c not in {"Patient_ID", "Diabetes_Type"}]
    sample = df.sample(min(20000, len(df)), random_state=42)
    X = sample[candidates]
    y = sample["Diabetes_Type"]
    rf = RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=candidates).sort_values(ascending=False)
    return importances.head(min(8, len(importances))).index.tolist()


@st.cache_resource(show_spinner=False)
def load_or_train_rf(path: str) -> Tuple[RandomForestClassifier, List[str], Dict[str, float]]:
    df = load_diabetes_data(path)
    features = load_selected_features(df)

    if PRETRAINED_RF.exists():
        try:
            model = joblib.load(PRETRAINED_RF)
            return model, features, {"source": "pretrained", "accuracy": float("nan")}
        except Exception:
            pass

    X = df[features]
    y = df["Diabetes_Type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=260,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, features, {"source": "trained in app", "accuracy": accuracy_score(y_test, pred)}


def rf_importance_table(model: RandomForestClassifier, features: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_") and len(model.feature_importances_) == len(features):
        scores = model.feature_importances_
    else:
        scores = np.ones(len(features)) / max(len(features), 1)
    table = pd.DataFrame({"Feature": features, "Importance score": scores})
    return table.sort_values("Importance score", ascending=False).reset_index(drop=True)


def diagnosis_validation(inputs: Dict[str, float]) -> List[str]:
    checks = []
    if inputs["BMI"] < 12 or inputs["BMI"] > 60:
        checks.append("BMI is outside the expected adult screening range.")
    if inputs["HbA1c"] >= 6.5 or inputs["Fasting_Glucose"] >= 126 or inputs["Random_Glucose"] >= 200:
        checks.append("Glucose/HbA1c values meet common diabetes screening thresholds.")
    if inputs["GAD_Antibody"] == 1 and inputs["C_Peptide"] < 1.0:
        checks.append("Positive GAD antibody with low C-peptide supports autoimmune/type 1 pattern.")
    if inputs["Age"] < 1 or inputs["Age"] > 110:
        checks.append("Age is outside expected input bounds.")
    if inputs.get("Weight_kg", 1) <= 0:
        checks.append("Weight must be positive.")
    if "Height_cm" in inputs and inputs["Height_cm"] <= 0:
        checks.append("Height must be positive.")
    return checks or ["Structured validation passed."]


def gemini_validate(payload: Dict) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    prompt = (
        "Validate this diabetes diagnosis and insulin recommendation as a clinical safety reviewer. "
        "Do not invent a diagnosis. Return concise JSON with fields risk_level, concerns, and advice. "
        f"Payload: {json.dumps(payload, default=str)}"
    )
    body = json.dumps(
        {"contents": [{"parts": [{"text": prompt}]}]},
    ).encode("utf-8")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    req = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (URLError, KeyError, TimeoutError, ValueError):
        return None


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%d-%m-%Y %H:%M:%S")


def integrate_basal_units(events: List[Dict]) -> float:
    total = 0.0
    sorted_events = sorted(events, key=lambda e: e["ts"])
    for i, event in enumerate(sorted_events):
        start = event["ts"]
        end = sorted_events[i + 1]["ts"] if i + 1 < len(sorted_events) else start.replace(hour=23, minute=59)
        hours = max((end - start).total_seconds() / 3600, 0)
        total += event["value"] * hours
    return total


@st.cache_data(show_spinner=False)
def load_cgm_patient_days(zip_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    patient_rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml") or "training" not in name:
                continue
            root = ET.fromstring(zf.read(name))
            patient_id = root.attrib.get("id", name.split("-")[0])
            weight = float(root.attrib.get("weight", 70))
            glucose = []
            basal_events_by_day: Dict[str, List[Dict]] = {}
            bolus_by_day: Dict[str, float] = {}

            for event in root.find("glucose_level") or []:
                ts = parse_ts(event.attrib["ts"])
                glucose.append({"Patient_ID": patient_id, "ts": ts, "date": ts.date(), "glucose": float(event.attrib["value"])})

            for event in root.find("basal") or []:
                ts = parse_ts(event.attrib["ts"])
                basal_events_by_day.setdefault(str(ts.date()), []).append({"ts": ts, "value": float(event.attrib["value"])})

            for event in root.find("bolus") or []:
                ts = parse_ts(event.attrib.get("ts_begin") or event.attrib.get("ts"))
                bolus_by_day[str(ts.date())] = bolus_by_day.get(str(ts.date()), 0.0) + float(event.attrib.get("dose", 0))

            gdf = pd.DataFrame(glucose)
            if gdf.empty:
                continue
            patient_rows.append({"CGM_Patient_ID": patient_id, "Weight_kg": weight, "Records": len(gdf)})

            for date, day in gdf.groupby("date"):
                values = day.sort_values("ts")["glucose"].to_numpy(dtype=float)
                if len(values) < 48:
                    continue
                sampled = np.interp(np.linspace(0, len(values) - 1, 96), np.arange(len(values)), values)
                basal = integrate_basal_units(basal_events_by_day.get(str(date), []))
                bolus = bolus_by_day.get(str(date), 0.0)
                if basal <= 0:
                    basal = max(weight * 0.25, 1.0)
                if bolus <= 0:
                    bolus = max(weight * 0.20, 1.0)
                rows.append(
                    {
                        "CGM_Patient_ID": patient_id,
                        "date": str(date),
                        "weight": weight,
                        "sequence": sampled.astype(np.float32),
                        "mean_glucose": float(np.mean(values)),
                        "time_in_range": float(np.mean((values >= 70) & (values <= 180))),
                        "time_high": float(np.mean(values > 180)),
                        "time_low": float(np.mean(values < 70)),
                        "basal_units": float(basal),
                        "bolus_units": float(bolus),
                        "tdd_units": float(basal + bolus),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(patient_rows).drop_duplicates("CGM_Patient_ID")


@st.cache_resource(show_spinner=False)
def train_lstm(zip_path: str) -> LSTMArtifacts:
    day_df, patient_table = load_cgm_patient_days(zip_path)
    if day_df.empty:
        raise ValueError("No CGM patient-day records found.")

    sequences = np.stack(day_df["sequence"].to_numpy())
    mean = float(sequences.mean())
    std = float(sequences.std() or 1.0)
    X = ((sequences - mean) / std)[:, :, None].astype(np.float32)
    y = day_df[["tdd_units", "basal_units", "bolus_units"]].to_numpy(dtype=np.float32)

    model = DoseLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.SmoothL1Loss()
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    for _ in range(220):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(Xt).numpy()
    mae = float(np.mean(np.abs(pred - y)))
    return LSTMArtifacts(
        model=model.eval(),
        scaler_mean=mean,
        scaler_std=std,
        patient_table=patient_table,
        training_summary={"patient_days": float(len(day_df)), "mae_units": mae},
    )


def type1_lstm_recommendation(artifacts: LSTMArtifacts, cgm_zip: str, cgm_patient_id: str) -> Dict:
    day_df, _ = load_cgm_patient_days(cgm_zip)
    patient_days = day_df[day_df["CGM_Patient_ID"] == cgm_patient_id].copy()
    if patient_days.empty:
        patient_days = day_df.copy()
    latest = patient_days.sort_values("date").iloc[-1]
    seq = ((latest["sequence"] - artifacts.scaler_mean) / artifacts.scaler_std)[None, :, None].astype(np.float32)
    with torch.no_grad():
        pred = artifacts.model(torch.tensor(seq)).numpy()[0]
    tdd, basal, bolus = [max(float(v), 0.0) for v in pred]
    if tdd > 0:
        scale = tdd / max(basal + bolus, 0.001)
        basal, bolus = basal * scale, bolus * scale
    return {
        "method": "PyTorch LSTM over 96-point CGM daily sequence",
        "cgm_patient_id": cgm_patient_id,
        "tdd": round(tdd, 1),
        "basal": round(basal, 1),
        "bolus": round(bolus, 1),
        "mean_glucose": round(float(latest["mean_glucose"]), 1),
        "time_in_range": round(float(latest["time_in_range"]) * 100, 1),
        "time_high": round(float(latest["time_high"]) * 100, 1),
        "time_low": round(float(latest["time_low"]) * 100, 1),
        "explainability": [
            f"Recent CGM mean glucose is {latest['mean_glucose']:.1f} mg/dL.",
            f"Time in range is {latest['time_in_range'] * 100:.1f}%; time high is {latest['time_high'] * 100:.1f}%.",
            "The LSTM uses the full daily glucose curve rather than a single fasting value.",
        ],
    }


def type2_formula_recommendation(weight: float, fasting_glucose: float, previous_tdd: float = 0.0) -> Dict:
    low = weight * 0.5
    high = weight * 0.6
    base_tdd = (low + high) / 2
    titration = 0
    if fasting_glucose > 180:
        titration = 6
    elif fasting_glucose > 130:
        titration = 3
    elif fasting_glucose < 80:
        titration = -3
    if previous_tdd > 0:
        base_tdd = previous_tdd
    tdd = max(base_tdd + titration, 0)
    basal = tdd * 0.5
    bolus = tdd * 0.5
    return {
        "method": "Clinical formula TDD = weight(kg) x 0.5-0.6 plus 3-3 titration",
        "formula_range": f"{low:.1f}-{high:.1f} units/day",
        "titration_adjustment": titration,
        "tdd": round(tdd, 1),
        "basal": round(basal, 1),
        "bolus": round(bolus, 1),
        "explainability": [
            f"Weight-based formula gives {low:.1f}-{high:.1f} units/day.",
            f"Fasting glucose of {fasting_glucose:.1f} mg/dL maps to a {titration:+d} unit titration step.",
            "Basal and bolus are split 50/50 because no meal-specific carbohydrate schedule is provided.",
        ],
    }


def insulin_validation(reco: Dict, diabetes_type: str, weight: float) -> List[str]:
    notes = []
    units_per_kg = reco["tdd"] / max(weight, 1)
    if diabetes_type == "Type 1 diabetes" and not 0.2 <= units_per_kg <= 1.2:
        notes.append(f"TDD is {units_per_kg:.2f} units/kg, outside the broad review band for type 1.")
    if diabetes_type == "Type 2 diabetes" and not 0.1 <= units_per_kg <= 1.2:
        notes.append(f"TDD is {units_per_kg:.2f} units/kg, outside the broad review band for type 2.")
    if reco.get("time_low", 0) > 4:
        notes.append("CGM shows elevated hypoglycemia exposure; dose should be clinically reviewed.")
    if reco["tdd"] <= 0:
        notes.append("Dose is non-positive and cannot be used.")
    return notes or ["Dose validation passed basic safety checks."]


def prediction_explanation(inputs: Dict[str, float], importance: pd.DataFrame, predicted: str) -> pd.DataFrame:
    rows = []
    for _, row in importance.iterrows():
        feature = row["Feature"]
        value = inputs.get(feature)
        if value is None:
            continue
        rows.append(
            {
                "Feature": feature,
                "Input value": value,
                "Importance score": row["Importance score"],
                "Clinical signal": clinical_signal(feature, value, predicted),
            }
        )
    return pd.DataFrame(rows)


def clinical_signal(feature: str, value: float, predicted: str) -> str:
    if feature == "HbA1c":
        return "Elevated" if value >= 6.5 else "Within screening range"
    if feature == "Fasting_Glucose":
        return "Elevated" if value >= 126 else "Below diagnostic threshold"
    if feature == "Random_Glucose":
        return "Elevated" if value >= 200 else "Below diagnostic threshold"
    if feature == "C_Peptide":
        return "Low insulin reserve signal" if value < 1.0 else "Preserved insulin reserve signal"
    if feature == "GAD_Antibody":
        return "Autoimmune marker present" if value == 1 else "Autoimmune marker absent"
    if feature == "BMI":
        return "Insulin resistance risk" if value >= 25 else "Lower BMI risk signal"
    return "Contributes through trained Random Forest split patterns"


st.set_page_config(page_title="Diabetes AI Diagnosis and Insulin Recommender", layout="wide")
st.title("Intelligent Diabetes Diagnosis and Insulin Recommendation")
st.caption("Random Forest diagnosis + confidence threshold + validation layer + LSTM/formula insulin recommendation")

if not DEFAULT_DIABETES_CSV.exists():
    st.error(f"Diabetes dataset not found: {DEFAULT_DIABETES_CSV}")
    st.stop()
if not DEFAULT_CGM_ZIP.exists():
    st.error(f"CGM archive not found: {DEFAULT_CGM_ZIP}")
    st.stop()

df = load_diabetes_data(str(DEFAULT_DIABETES_CSV))
model, selected_features, rf_summary = load_or_train_rf(str(DEFAULT_DIABETES_CSV))
importance = rf_importance_table(model, selected_features)

tabs = st.tabs(["Patient Diagnosis", "Insulin Recommendation", "Model Details"])

with tabs[0]:
    st.subheader("Patient Input")
    left, right = st.columns([1.2, 1])
    with left:
        source = st.radio("Input mode", ["Manual patient", "Fetch patient from dataset"], horizontal=True)
        selected_patient = None
        if source == "Fetch patient from dataset":
            options = df["Patient_ID"].head(5000).tolist()
            selected_patient = st.selectbox("Patient ID", options, index=0)
            row = df.loc[df["Patient_ID"] == selected_patient].iloc[0]
            defaults = {feature: row[feature] for feature in selected_features}
        else:
            defaults = {
                "Age": 45,
                "Weight_kg": 72.0,
                "Height_cm": 165,
                "HbA1c": 7.2,
                "Fasting_Glucose": 135.0,
                "Random_Glucose": 190.0,
                "C_Peptide": 2.0,
                "GAD_Antibody": 0,
                "BMI": 26.4,
            }

        inputs = {}
        c1, c2 = st.columns(2)
        for i, feature in enumerate(selected_features):
            target_col = c1 if i % 2 == 0 else c2
            if feature == "GAD_Antibody":
                inputs[feature] = target_col.selectbox(feature, [0, 1], index=int(defaults.get(feature, 0)))
            elif feature in {"Age", "Height_cm"}:
                inputs[feature] = target_col.number_input(feature, value=int(defaults.get(feature, 0)), step=1)
            else:
                inputs[feature] = target_col.number_input(feature, value=float(defaults.get(feature, 0)), step=0.1)

        threshold = st.slider("Confidence threshold", 0.50, 0.95, 0.72, 0.01)

    with right:
        X = pd.DataFrame([{feature: inputs[feature] for feature in selected_features}])
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        best_idx = int(np.argmax(probs))
        predicted_code = int(classes[best_idx])
        confidence = float(probs[best_idx])
        predicted_label = CLASS_NAMES.get(predicted_code, str(predicted_code))
        decision = predicted_label if confidence >= threshold else "Needs clinician review"

        st.metric("Diagnosis decision", decision)
        st.metric("Model confidence", f"{confidence * 100:.1f}%")
        if selected_patient is None and predicted_label in DIABETIC_CLASSES:
            selected_patient = patient_id_for_index(abs(hash(tuple(inputs.values()))) % 900000)
        if predicted_label in DIABETIC_CLASSES and confidence >= threshold:
            st.success(f"Diabetic patient automatically routed to recommender: {selected_patient}")
        elif predicted_label in DIABETIC_CLASSES:
            st.warning("Diabetic pattern detected, but confidence is below threshold.")
        else:
            st.info("No insulin recommendation is routed for non-diabetic decision.")

        prob_df = pd.DataFrame(
            {"Class": [CLASS_NAMES.get(int(c), str(c)) for c in classes], "Probability": probs}
        )
        st.plotly_chart(px.bar(prob_df, x="Class", y="Probability", range_y=[0, 1]), width="stretch")

    st.subheader("Validation Layer")
    for note in diagnosis_validation(inputs):
        st.write(f"- {note}")

    st.subheader("Explainability")
    st.dataframe(prediction_explanation(inputs, importance, predicted_label), width="stretch")

    st.session_state["latest_patient"] = {
        "patient_id": selected_patient,
        "inputs": inputs,
        "diagnosis": predicted_label,
        "decision": decision,
        "confidence": confidence,
        "threshold": threshold,
    }

with tabs[1]:
    st.subheader("Insulin Recommender")
    latest = st.session_state.get("latest_patient")
    if not latest:
        st.info("Run a diagnosis first to auto-fetch a diabetic patient into this recommender.")
    else:
        st.write(f"Patient ID: **{latest['patient_id']}**")
        st.write(f"Diagnosis route: **{latest['diagnosis']}** at **{latest['confidence'] * 100:.1f}%** confidence")
        if latest["decision"] == "Needs clinician review":
            st.warning("Recommendation is paused because diagnosis confidence is below the selected threshold.")
        elif latest["diagnosis"] == "Non-diabetic":
            st.info("No insulin recommendation is generated for a non-diabetic classification.")
        else:
            weight = float(latest["inputs"].get("Weight_kg", 70))
            if latest["diagnosis"] == "Type 1 diabetes":
                with st.spinner("Training/loading LSTM and reading CGM data..."):
                    lstm_artifacts = train_lstm(str(DEFAULT_CGM_ZIP))
                cgm_options = lstm_artifacts.patient_table["CGM_Patient_ID"].astype(str).tolist()
                cgm_patient = st.selectbox("CGM patient stream for Type 1 LSTM", cgm_options)
                reco = type1_lstm_recommendation(lstm_artifacts, str(DEFAULT_CGM_ZIP), cgm_patient)
                st.caption(
                    f"LSTM trained on {int(lstm_artifacts.training_summary['patient_days'])} CGM patient-days; "
                    f"training MAE {lstm_artifacts.training_summary['mae_units']:.2f} units."
                )
            else:
                previous_tdd = st.number_input("Previous total daily insulin dose, if available", value=0.0, step=1.0)
                reco = type2_formula_recommendation(
                    weight=weight,
                    fasting_glucose=float(latest["inputs"].get("Fasting_Glucose", 110)),
                    previous_tdd=previous_tdd,
                )

            a, b, c = st.columns(3)
            a.metric("Total daily dose", f"{reco['tdd']:.1f} U/day")
            b.metric("Basal", f"{reco['basal']:.1f} U/day")
            c.metric("Bolus", f"{reco['bolus']:.1f} U/day")
            st.write(f"Method: **{reco['method']}**")
            if "formula_range" in reco:
                st.write(f"Formula range: **{reco['formula_range']}**")
                st.write(f"3-3 titration adjustment: **{reco['titration_adjustment']:+d} units**")

            st.subheader("Insulin Validation Layer")
            validation_notes = insulin_validation(reco, latest["diagnosis"], weight)
            for note in validation_notes:
                st.write(f"- {note}")

            st.subheader("Recommendation Explainability")
            for note in reco["explainability"]:
                st.write(f"- {note}")

            llm_payload = {"patient": latest, "recommendation": reco, "validation": validation_notes}
            llm_result = gemini_validate(llm_payload)
            with st.expander("LLM validation layer"):
                if llm_result:
                    st.write(llm_result)
                else:
                    st.write(
                        "Gemini validation is ready when GEMINI_API_KEY is configured. "
                        "For this local run, the app used the deterministic clinical validation checks above."
                    )

with tabs[2]:
    st.subheader("Selected Diagnosis Features")
    st.dataframe(importance, width="stretch")
    st.subheader("Dataset Summary")
    st.write(f"Rows: **{len(df):,}**")
    st.write(f"Random Forest source: **{rf_summary['source']}**")
    if not math.isnan(rf_summary.get("accuracy", float("nan"))):
        st.write(f"Holdout accuracy: **{rf_summary['accuracy'] * 100:.2f}%**")
    st.write("Class labels: 0 = Non-diabetic, 1 = Type 1 diabetes, 2 = Type 2 diabetes")
    st.subheader("Clinical Safety Note")
    st.warning(
        "This project app is for research/prototype use only. Insulin decisions require licensed clinical review, "
        "local protocols, medication history, renal status, meals, hypoglycemia history, and patient-specific targets."
    )

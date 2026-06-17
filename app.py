import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request as urlrequest
from urllib.error import URLError
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DIABETES_CSV = Path(os.getenv("DIABETES_DATASET", DATA_DIR / "diabetes_realistic_dataset.csv"))
CGM_ZIP = Path(os.getenv("CGM_ARCHIVE", DATA_DIR / "cgm_archive.zip"))

CLASS_NAMES = {
    0: "Non-diabetic",
    1: "Type 1 diabetes",
    2: "Type 2 diabetes",
}

SELECTED_FEATURES = [
    "HbA1c",
    "Fasting_Glucose",
    "C_Peptide",
    "Age",
    "GAD_Antibody",
    "BMI",
    "Weight_kg",
]


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
    mean: float
    std: float
    patient_table: pd.DataFrame
    day_table: pd.DataFrame
    mae_units: float


def patient_id(index: int) -> str:
    return f"PT-{index + 1:06d}"


@st.cache_data(show_spinner=False)
def load_diabetes_data() -> pd.DataFrame:
    df = pd.read_csv(DIABETES_CSV)
    df = df.copy()
    df.insert(0, "Patient_ID", [patient_id(i) for i in range(len(df))])
    return df


@st.cache_resource(show_spinner=False)
def train_random_forest() -> Tuple[RandomForestClassifier, float, pd.DataFrame]:
    df = load_diabetes_data()
    X = df[SELECTED_FEATURES]
    y = df["Diabetes_Type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    importance = pd.DataFrame(
        {"Feature": SELECTED_FEATURES, "Importance score": model.feature_importances_}
    ).sort_values("Importance score", ascending=False)
    return model, accuracy, importance


def diagnosis_validation(values: Dict[str, float]) -> List[str]:
    notes = []
    if values["HbA1c"] >= 6.5 or values["Fasting_Glucose"] >= 126:
        notes.append("HbA1c or fasting glucose crosses common diabetes screening thresholds.")
    if values["GAD_Antibody"] == 1 and values["C_Peptide"] < 1.0:
        notes.append("Positive GAD antibody with low C-peptide supports a type 1 pattern.")
    if values["BMI"] < 12 or values["BMI"] > 60:
        notes.append("BMI is outside the usual adult screening range.")
    if values["Age"] < 1 or values["Age"] > 110:
        notes.append("Age is outside expected input bounds.")
    if values["Weight_kg"] <= 0:
        notes.append("Weight must be positive.")
    return notes or ["Structured validation passed."]


def clinical_signal(feature: str, value: float) -> str:
    if feature == "HbA1c":
        return "Elevated" if value >= 6.5 else "Below diabetes threshold"
    if feature == "Fasting_Glucose":
        return "Elevated" if value >= 126 else "Below diabetes threshold"
    if feature == "C_Peptide":
        return "Low insulin reserve signal" if value < 1.0 else "Preserved insulin reserve signal"
    if feature == "GAD_Antibody":
        return "Autoimmune marker present" if value == 1 else "Autoimmune marker absent"
    if feature == "BMI":
        return "Insulin-resistance risk signal" if value >= 25 else "Lower BMI risk signal"
    return "Contributes through Random Forest split patterns"


def explain_prediction(values: Dict[str, float], importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in importance.iterrows():
        feature = row["Feature"]
        rows.append(
            {
                "Feature": feature,
                "Input value": values[feature],
                "Importance score": row["Importance score"],
                "Clinical signal": clinical_signal(feature, values[feature]),
            }
        )
    return pd.DataFrame(rows)


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%d-%m-%Y %H:%M:%S")


def integrate_basal(events: List[Dict]) -> float:
    total = 0.0
    events = sorted(events, key=lambda event: event["ts"])
    for index, event in enumerate(events):
        start = event["ts"]
        end = events[index + 1]["ts"] if index + 1 < len(events) else start.replace(hour=23, minute=59)
        total += event["value"] * max((end - start).total_seconds() / 3600, 0)
    return total


@st.cache_data(show_spinner=False)
def load_cgm_days() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    patients = []
    with zipfile.ZipFile(CGM_ZIP) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml") or "training" not in name:
                continue
            root = ET.fromstring(zf.read(name))
            cgm_patient_id = root.attrib.get("id", name.split("-")[0])
            weight = float(root.attrib.get("weight", 70))
            glucose_rows = []
            basal_by_day: Dict[str, List[Dict]] = {}
            bolus_by_day: Dict[str, float] = {}

            glucose_node = root.find("glucose_level")
            if glucose_node is not None:
                for event in glucose_node:
                    ts = parse_ts(event.attrib["ts"])
                    glucose_rows.append({"ts": ts, "date": ts.date(), "glucose": float(event.attrib["value"])})

            basal_node = root.find("basal")
            if basal_node is not None:
                for event in basal_node:
                    ts = parse_ts(event.attrib["ts"])
                    basal_by_day.setdefault(str(ts.date()), []).append(
                        {"ts": ts, "value": float(event.attrib["value"])}
                    )

            bolus_node = root.find("bolus")
            if bolus_node is not None:
                for event in bolus_node:
                    ts = parse_ts(event.attrib.get("ts_begin") or event.attrib.get("ts"))
                    bolus_by_day[str(ts.date())] = bolus_by_day.get(str(ts.date()), 0.0) + float(
                        event.attrib.get("dose", 0)
                    )

            glucose_df = pd.DataFrame(glucose_rows)
            if glucose_df.empty:
                continue
            patients.append({"CGM_Patient_ID": cgm_patient_id, "Weight_kg": weight, "Records": len(glucose_df)})

            for date, day in glucose_df.groupby("date"):
                values = day.sort_values("ts")["glucose"].to_numpy(dtype=float)
                if len(values) < 48:
                    continue
                sequence = np.interp(np.linspace(0, len(values) - 1, 96), np.arange(len(values)), values)
                basal = integrate_basal(basal_by_day.get(str(date), []))
                bolus = bolus_by_day.get(str(date), 0.0)
                if basal <= 0:
                    basal = max(weight * 0.25, 1.0)
                if bolus <= 0:
                    bolus = max(weight * 0.20, 1.0)
                rows.append(
                    {
                        "CGM_Patient_ID": cgm_patient_id,
                        "date": str(date),
                        "weight": weight,
                        "sequence": sequence.astype(np.float32),
                        "mean_glucose": float(np.mean(values)),
                        "time_in_range": float(np.mean((values >= 70) & (values <= 180))),
                        "time_high": float(np.mean(values > 180)),
                        "time_low": float(np.mean(values < 70)),
                        "tdd": float(basal + bolus),
                        "basal": float(basal),
                        "bolus": float(bolus),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(patients).drop_duplicates("CGM_Patient_ID")


@st.cache_resource(show_spinner=False)
def train_lstm() -> LSTMArtifacts:
    day_table, patient_table = load_cgm_days()
    sequences = np.stack(day_table["sequence"].to_numpy())
    y = day_table[["tdd", "basal", "bolus"]].to_numpy(dtype=np.float32)
    mean = float(sequences.mean())
    std = float(sequences.std() or 1)
    X = ((sequences - mean) / std)[:, :, None].astype(np.float32)

    model = DoseLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.SmoothL1Loss()
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    for _ in range(180):
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mae = float(np.mean(np.abs(model(Xt).numpy() - y)))
    return LSTMArtifacts(model.eval(), mean, std, patient_table, day_table, mae)


def type1_recommendation(artifacts: LSTMArtifacts, cgm_patient_id: str) -> Dict:
    rows = artifacts.day_table[artifacts.day_table["CGM_Patient_ID"] == cgm_patient_id].sort_values("date")
    latest = rows.iloc[-1] if not rows.empty else artifacts.day_table.iloc[-1]
    sequence = ((latest["sequence"] - artifacts.mean) / artifacts.std)[None, :, None].astype(np.float32)
    with torch.no_grad():
        tdd, basal, bolus = [max(float(value), 0.0) for value in artifacts.model(torch.tensor(sequence)).numpy()[0]]
    scale = tdd / max(basal + bolus, 0.001)
    basal *= scale
    bolus *= scale
    return {
        "method": "LSTM over 96-point daily CGM sequence",
        "tdd": round(tdd, 1),
        "basal": round(basal, 1),
        "bolus": round(bolus, 1),
        "mean_glucose": round(float(latest["mean_glucose"]), 1),
        "time_in_range": round(float(latest["time_in_range"]) * 100, 1),
        "time_high": round(float(latest["time_high"]) * 100, 1),
        "time_low": round(float(latest["time_low"]) * 100, 1),
    }


def type2_recommendation(weight: float, fasting_glucose: float, previous_tdd: float) -> Dict:
    low = weight * 0.5
    high = weight * 0.6
    base = previous_tdd if previous_tdd > 0 else (low + high) / 2
    titration = 0
    if fasting_glucose > 180:
        titration = 6
    elif fasting_glucose > 130:
        titration = 3
    elif fasting_glucose < 80:
        titration = -3
    tdd = max(base + titration, 0)
    return {
        "method": "Clinical formula TDD = weight(kg) x 0.5-0.6 with 3-3 titration",
        "formula_range": f"{low:.1f}-{high:.1f} units/day",
        "titration": titration,
        "tdd": round(tdd, 1),
        "basal": round(tdd * 0.5, 1),
        "bolus": round(tdd * 0.5, 1),
    }


def insulin_validation(reco: Dict, weight: float, diagnosis: str) -> List[str]:
    notes = []
    units_per_kg = reco["tdd"] / max(weight, 1)
    if units_per_kg > 1.2:
        notes.append(f"High-dose safety flag: {units_per_kg:.2f} units/kg/day.")
    if units_per_kg < 0.1:
        notes.append(f"Low-dose safety flag: {units_per_kg:.2f} units/kg/day.")
    if reco.get("time_low", 0) > 4:
        notes.append("CGM shows more than 4% time below range; clinical review is required.")
    if diagnosis == "Type 2 diabetes" and reco["titration"] != 0:
        notes.append(f"3-3 titration adjusted TDD by {reco['titration']:+d} units.")
    return notes or ["Dose validation passed basic safety checks."]


def gemini_validate(payload: Dict) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    prompt = (
        "Validate this diabetes diagnosis and insulin recommendation as a concise clinical safety reviewer. "
        "Return JSON with risk_level, concerns, and advice. "
        f"Payload: {json.dumps(payload, default=str)}"
    )
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    req = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (URLError, KeyError, TimeoutError, ValueError):
        return None


st.set_page_config(page_title="Diabetes Diagnosis and Insulin Recommendation", layout="wide")
st.title("Intelligent Diabetes Diagnosis and Insulin Recommendation")
st.caption("Random Forest diagnosis + confidence threshold + validation + LSTM/formula insulin recommendation")

if not DIABETES_CSV.exists():
    st.error(f"Missing diabetes dataset: {DIABETES_CSV}")
    st.stop()
if not CGM_ZIP.exists():
    st.error(f"Missing CGM archive: {CGM_ZIP}")
    st.stop()

with st.spinner("Loading dataset and training Random Forest classifier..."):
    diabetes_df = load_diabetes_data()
    rf_model, rf_accuracy, importance_table = train_random_forest()

tabs = st.tabs(["Diagnosis", "Insulin Recommendation", "Model Details"])

with tabs[0]:
    st.subheader("Patient Diagnosis")
    left, right = st.columns([1.2, 1])
    with left:
        mode = st.radio("Input mode", ["Manual patient", "Fetch patient from dataset"], horizontal=True)
        selected_patient = None
        if mode == "Fetch patient from dataset":
            selected_patient = st.selectbox("Patient ID", diabetes_df["Patient_ID"].head(5000).tolist())
            selected_row = diabetes_df.loc[diabetes_df["Patient_ID"] == selected_patient].iloc[0]
            defaults = {feature: selected_row[feature] for feature in SELECTED_FEATURES}
        else:
            defaults = {
                "HbA1c": 7.2,
                "Fasting_Glucose": 135.0,
                "C_Peptide": 2.0,
                "Age": 45,
                "GAD_Antibody": 0,
                "BMI": 26.4,
                "Weight_kg": 72.0,
            }

        values: Dict[str, float] = {}
        c1, c2 = st.columns(2)
        for index, feature in enumerate(SELECTED_FEATURES):
            column = c1 if index % 2 == 0 else c2
            if feature == "GAD_Antibody":
                values[feature] = column.selectbox(feature, [0, 1], index=int(defaults[feature]))
            elif feature == "Age":
                values[feature] = column.number_input(feature, value=int(defaults[feature]), min_value=1, max_value=110)
            else:
                values[feature] = column.number_input(feature, value=float(defaults[feature]), step=0.1)
        confidence_threshold = st.slider("Confidence threshold", 0.50, 0.95, 0.72, 0.01)

    with right:
        X = pd.DataFrame([{feature: values[feature] for feature in SELECTED_FEATURES}])
        probabilities = rf_model.predict_proba(X)[0]
        classes = list(rf_model.classes_)
        best_index = int(np.argmax(probabilities))
        predicted_code = int(classes[best_index])
        predicted_label = CLASS_NAMES[predicted_code]
        confidence = float(probabilities[best_index])
        decision = predicted_label if confidence >= confidence_threshold else "Needs clinician review"

        st.metric("Diagnosis decision", decision)
        st.metric("Model confidence", f"{confidence * 100:.1f}%")
        if selected_patient is None:
            selected_patient = patient_id(abs(hash(tuple(values.values()))) % 900000)
        if predicted_label != "Non-diabetic" and confidence >= confidence_threshold:
            st.success(f"Patient routed to insulin recommender: {selected_patient}")
        elif predicted_label != "Non-diabetic":
            st.warning("Diabetic pattern detected, but confidence is below threshold.")
        else:
            st.info("No insulin recommendation is routed for non-diabetic classification.")

        probability_table = pd.DataFrame(
            {"Class": [CLASS_NAMES[int(code)] for code in classes], "Probability": probabilities}
        )
        st.plotly_chart(px.bar(probability_table, x="Class", y="Probability", range_y=[0, 1]), width="stretch")

    st.subheader("Validation Layer")
    for note in diagnosis_validation(values):
        st.write(f"- {note}")

    st.subheader("Explainability")
    st.dataframe(explain_prediction(values, importance_table), width="stretch")

    st.session_state["latest_patient"] = {
        "patient_id": selected_patient,
        "values": values,
        "diagnosis": predicted_label,
        "decision": decision,
        "confidence": confidence,
        "threshold": confidence_threshold,
    }

with tabs[1]:
    st.subheader("Insulin Recommendation")
    latest = st.session_state.get("latest_patient")
    if not latest:
        st.info("Run a diagnosis first. Diabetic patients are automatically routed here.")
    elif latest["decision"] == "Needs clinician review":
        st.warning("Recommendation paused because diagnosis confidence is below the threshold.")
    elif latest["diagnosis"] == "Non-diabetic":
        st.info("No insulin recommendation is generated for a non-diabetic result.")
    else:
        st.write(f"Patient ID: **{latest['patient_id']}**")
        st.write(f"Diagnosis: **{latest['diagnosis']}**")
        weight = float(latest["values"]["Weight_kg"])
        if latest["diagnosis"] == "Type 1 diabetes":
            with st.spinner("Training LSTM on CGM patient-day sequences..."):
                artifacts = train_lstm()
            cgm_patient = st.selectbox(
                "CGM patient stream",
                artifacts.patient_table["CGM_Patient_ID"].astype(str).tolist(),
            )
            reco = type1_recommendation(artifacts, cgm_patient)
            st.caption(f"LSTM training MAE: {artifacts.mae_units:.2f} insulin units")
        else:
            previous_tdd = st.number_input("Previous total daily dose, if available", value=0.0, step=1.0)
            reco = type2_recommendation(weight, float(latest["values"]["Fasting_Glucose"]), previous_tdd)

        a, b, c = st.columns(3)
        a.metric("Total Daily Dose", f"{reco['tdd']:.1f} U/day")
        b.metric("Basal", f"{reco['basal']:.1f} U/day")
        c.metric("Bolus", f"{reco['bolus']:.1f} U/day")
        st.write(f"Method: **{reco['method']}**")
        if "formula_range" in reco:
            st.write(f"Formula range: **{reco['formula_range']}**")
            st.write(f"3-3 titration adjustment: **{reco['titration']:+d} units**")

        st.subheader("Insulin Validation Layer")
        validation_notes = insulin_validation(reco, weight, latest["diagnosis"])
        for note in validation_notes:
            st.write(f"- {note}")

        st.subheader("Recommendation Explainability")
        if latest["diagnosis"] == "Type 1 diabetes":
            st.write(f"- Recent CGM mean glucose: {reco['mean_glucose']} mg/dL")
            st.write(f"- Time in range: {reco['time_in_range']}%")
            st.write(f"- Time high: {reco['time_high']}%")
            st.write("- The LSTM uses the daily CGM curve to estimate total, basal, and bolus dose.")
        else:
            st.write("- Type 2 recommendation uses the clinical weight-based TDD formula.")
            st.write("- The 3-3 titration adjustment is driven by fasting glucose.")
            st.write("- Basal and bolus are split 50/50 because meal-specific carbohydrate data is not provided.")

        with st.expander("Gemini validation layer"):
            llm_response = gemini_validate({"patient": latest, "recommendation": reco, "validation": validation_notes})
            if llm_response:
                st.write(llm_response)
            else:
                st.write("Gemini validation is optional. Add GEMINI_API_KEY in Streamlit Secrets to enable it.")

with tabs[2]:
    st.subheader("Selected Classification Features")
    st.dataframe(importance_table, width="stretch")
    st.write(f"Random Forest holdout accuracy: **{rf_accuracy * 100:.2f}%**")
    st.write("Class labels: 0 = Non-diabetic, 1 = Type 1 diabetes, 2 = Type 2 diabetes")
    st.info("Random_Glucose was removed because leave-one-feature-out testing showed no accuracy loss.")
    st.warning(
        "Research/prototype use only. Insulin decisions require licensed clinical review and patient-specific context."
    )

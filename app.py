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

SELECTED_CLASSIFICATION_FEATURES = [
    "HbA1c",
    "Fasting_Glucose",
    "C_Peptide",
    "Age",
    "GAD_Antibody",
    "BMI",
    "Weight_kg",
]

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

# Intelligent Diabetes Diagnosis and Insulin Recommendation

Local Streamlit prototype for:

- Random Forest diabetes classification
- Confidence threshold gating
- Validation layer for diagnosis and insulin recommendations
- Explainability based on selected feature importance and clinical signals
- Type 1 insulin recommendation with a PyTorch LSTM over CGM sequences
- Type 2 insulin recommendation with the clinical formula `TDD = weight(kg) x 0.5-0.6 units/kg` plus 3-3 titration
- Optional Gemini validation layer through `GEMINI_API_KEY`

## Run

```powershell
python -m streamlit run app.py --server.port 8502
```

The app reads:

- `data/diabetes_realistic_dataset.csv`
- `data/cgm_archive.zip`
- Optional model artifacts when available through environment variables

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these project files to the repository:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `data/diabetes_realistic_dataset.csv`
   - `data/cgm_archive.zip`
3. Go to Streamlit Community Cloud and choose **New app**.
4. Select your GitHub repository, branch, and set the main file path to `app.py`.
5. Click **Deploy**.

Optional Gemini validation:

- In Streamlit Cloud, open **App settings > Secrets**.
- Add:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

The app still works without this key by using local validation rules.

## Selected Classification Features

The current selected feature file contains:

1. `HbA1c`
2. `Fasting_Glucose`
3. `C_Peptide`
4. `Age`
5. `GAD_Antibody`
6. `BMI`
7. `Random_Glucose`
8. `Weight_kg`

## Safety

This is a project prototype, not a medical device. All insulin outputs require clinician review.

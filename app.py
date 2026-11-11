import os

import boto3
import joblib
import pandas as pd
import streamlit as st


REGION = "ap-southeast-2"
MODEL_BUCKET = "mustafiz-heart-clean-new-2026"
MODEL_KEY = "random_forest_model.pkl"
LOCAL_MODEL_PATH = "random_forest_model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        s3 = boto3.client("s3", region_name=REGION)
        s3.download_file(MODEL_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
    return joblib.load(LOCAL_MODEL_PATH)


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("Heart Disease Prediction")
st.caption("Educational machine-learning demonstration; this is not medical advice.")

model = load_model()

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
    cp = st.selectbox("Chest pain type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol (chol)", min_value=50, max_value=700, value=200)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting ECG result (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum heart rate (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("ST slope (slope)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (thal)", [0, 1, 2, 3])
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=[
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ],
    )
    prediction = int(model.predict(input_data)[0])

    if prediction == 1:
        st.warning("Model output: 1 (possible heart-disease class in this dataset).")
    else:
        st.success("Model output: 0 (non-heart-disease class in this dataset).")

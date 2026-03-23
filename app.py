import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

model = pickle.load(open("KMean(2).pkl", "rb"))

df = pd.read_csv("heart.csv")

# Drop target column
feature_cols = [c for c in df.columns if c != "HeartDisease"]
df_features = df[feature_cols].copy()

# Encode text columns — fit encoders and save them
encoders = {}
text_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
for col in text_cols:
    le = LabelEncoder()
    df_features[col] = le.fit_transform(df_features[col])
    encoders[col] = le   # save each encoder for later use on user input

X = df_features.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca.fit(X_scaled)

st.title("❤️ Heart Disease Cluster Predictor")
st.write("Enter patient details below to find which cluster they belong to.")

age         = st.number_input("Age", 20, 100, 50)
sex         = st.selectbox("Sex", ["M", "F"])
cp          = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp  = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 0, 600, 200)
fasting_bs  = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr      = st.number_input("Max Heart Rate", 60, 220, 150)
ex_angina   = st.selectbox("Exercise Induced Angina", ["N", "Y"])
oldpeak     = st.number_input("Oldpeak (ST Depression)", -3.0, 10.0, 1.0)
st_slope    = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict Cluster"):

    # Step 1: Build raw input as DataFrame (same column order as training)
    raw_dict = {
        "Age":            [age],
        "Sex":            [sex],
        "ChestPainType":  [cp],
        "RestingBP":      [resting_bp],
        "Cholesterol":    [cholesterol],
        "FastingBS":      [fasting_bs],
        "RestingECG":     [resting_ecg],
        "MaxHR":          [max_hr],
        "ExerciseAngina": [ex_angina],
        "Oldpeak":        [oldpeak],
        "ST_Slope":       [st_slope],
    }
    input_df = pd.DataFrame(raw_dict)

    # Step 2: Encode text columns using the SAME encoders fitted on training data
    for col in text_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    # Step 3: Scale
    scaled_input = scaler.transform(input_df.values)

    # Step 4: PCA → 2 features
    pca_input = pca.transform(scaled_input)

    # Step 5: Predict cluster
    cluster = model.predict(pca_input)[0]

    st.success(f"✅ This patient belongs to **Cluster {cluster}**")

    if cluster == 0:
        st.info("🟢 Cluster 0: Lower risk group")
    else:
        st.warning("🔴 Cluster 1: Higher risk group")

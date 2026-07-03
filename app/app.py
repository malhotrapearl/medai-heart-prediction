import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import shap
from openai import OpenAI
from fpdf import FPDF

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="MedAI",
    page_icon="❤️",
    layout="wide"
)

# ================= API KEY =================
# 🔑 PUT YOUR KEY HERE
client = OpenAI(api_key="sk-proj-ym8YhhwGFPMvAGyrZkf3BGIkULwSLtvZXZT4jq-do3dK6LChZheJYZjvkbydkkjPTDzKcRnJmmT3BlbkFJISOkXIrvKB2y6HJqyIh9s5RtS_YoczyYcEH3ZTXk-T4skYavZziqgBE0JIXAnkydS6QeAbcg8AYE API_KEY")

# ================= CHAT HISTORY =================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= MODEL LOAD =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "heart_disease_xgb.pkl")
model = joblib.load(MODEL_PATH)

# ================= HEADER =================
st.markdown("""
<div style="
padding:25px;
border-radius:15px;
background:linear-gradient(135deg,#0f172a,#1e293b);
text-align:center;
color:white;
font-size:28px;
font-weight:bold;
">
❤️ MedAI - Heart Disease AI Doctor
</div>
""", unsafe_allow_html=True)

st.markdown("---")

risk_map = {
    0: "🟢 No Heart Disease",
    1: "🟡 Mild Heart Disease",
    2: "🟠 Moderate Heart Disease",
    3: "🔴 Severe Heart Disease",
    4: "⚫ Very Severe Heart Disease"
}

col1, col2 = st.columns(2)

# ================= INPUT =================
with col1:
    st.subheader("👤 Patient Info")

    dataset = st.selectbox(
        "Dataset",
        ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    )

    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Gender", ["Male", "Female"])
    cp = st.selectbox("Chest Pain", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.number_input("BP", 80, 250, 120)
    chol = st.number_input("Cholesterol", 100, 700, 220)
    fbs = st.selectbox("FBS", [True, False])
    restecg = st.selectbox("ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
    thalch = st.number_input("Max HR", 50, 250, 150)
    exang = st.selectbox("Exercise Angina", [True, False])
    oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("CA", 0, 4, 0)
    thal = st.selectbox("Thal", ["normal", "fixed defect", "reversible defect"])


# ================= PREDICTION =================
with col2:
    st.subheader("❤️ Prediction")

    if st.button("Predict"):

        patient = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "dataset": dataset,   # ✔ REQUIRED COLUMN
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalch": thalch,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        # 🔥 ALIGN COLUMNS (IMPORTANT FIX)
        patient = patient.reindex(
            columns=model.named_steps["preprocessor"].feature_names_in_,
            fill_value=0
        )

        prediction = model.predict(patient)
        probability = model.predict_proba(patient)

        risk = risk_map[int(prediction[0])]
        confidence = probability.max() * 100


        st.success(risk)
        st.metric("Confidence", f"{confidence:.2f}%")

        # Chart
        prob_df = pd.DataFrame({
            "Class": ["No", "Mild", "Moderate", "Severe", "Very Severe"],
            "Probability": probability[0]
        })

        st.bar_chart(prob_df.set_index("Class"))
        st.progress(int(prediction[0]) * 25)
# ================= CHATBOT =================
st.sidebar.title("🤖 MedAI Chatbot")

user_input = st.sidebar.chat_input("Ask about heart health...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful heart disease medical assistant."},
                *st.session_state.chat_history
            ]
        )
        reply = response.choices[0].message.content

    except:
        reply = "⚠ Invalid API Key or no internet connection."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.sidebar.info(reply)
    st.write(model.named_steps["preprocessor"].feature_names_in_)
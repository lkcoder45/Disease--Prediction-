import os
import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(MODEL_DIR, "disease_model.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    symptoms = joblib.load(os.path.join(MODEL_DIR, "symptoms.pkl"))
    return model, encoder, symptoms

st.title("🩺 Disease Prediction System")
st.caption("Machine Learning project — educational/demo use only")

try:
    model, encoder, symptoms = load_assets()
except FileNotFoundError:
    st.error("Model files not found. First run: python train_model.py")
    st.stop()

st.sidebar.header("Select Symptoms")
selected = st.multiselect(
    "Choose the symptoms you are experiencing:",
    options=symptoms,
    format_func=lambda x: x.replace("_", " ").title()
)

if st.button("🔍 Predict Disease", type="primary"):
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        row = pd.DataFrame([[int(s in selected) for s in symptoms]], columns=symptoms)
        probabilities = model.predict_proba(row)[0]
        order = probabilities.argsort()[::-1]
        top = order[:5]

        predicted = encoder.inverse_transform([top[0]])[0]
        confidence = probabilities[top[0]]

        st.success(f"Predicted Disease: **{predicted}**")
        st.metric("Model Confidence", f"{confidence:.2%}")

        st.subheader("Top Predictions")
        chart = pd.DataFrame({
            "Disease": encoder.inverse_transform(top),
            "Probability": probabilities[top]
        }).set_index("Disease")
        st.bar_chart(chart)

        st.info(
            "This is an educational machine-learning demonstration and is not a medical diagnosis. "
            "For real symptoms or emergencies, consult a qualified healthcare professional."
        )

st.divider()
st.markdown("### Project Features")
c1, c2, c3, c4 = st.columns(4)
c1.metric("ML Type", "Multiclass")
c2.metric("Algorithm", "Random Forest*")
c3.metric("Features", len(symptoms))
c4.metric("Classes", len(encoder.classes_))
st.caption("*The training script automatically selects the best-performing model.")

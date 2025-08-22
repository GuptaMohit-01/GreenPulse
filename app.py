import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- Page Config
st.set_page_config(page_title="GreenPulse", page_icon="🌱", layout="wide")

# --- CSS Styling
st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #f7fff7 0%, #ffffff 60%); }
    .card { background: white; border-radius: 12px; padding: 18px; 
            box-shadow: 0 6px 18px rgba(18, 66, 32, 0.08); }
    .big-metric { font-size: 20px; font-weight: 700; color: #065f46; }
    .muted { color: #6b7280; }
    .stButton>button {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white; border: none; padding: 10px 18px; 
        border-radius: 10px; font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header
col1, col2 = st.columns([1, 3])
with col1:
    try:
        st.image("docs/gif_logo.gif", width=120)
    except:
        st.image("docs/GreenPulse.png", width=120)
with col2:
    st.markdown("<h1>🌱 GreenPulse</h1><p class='muted'>Empowering Farmers with Smart AI Crop Recommendations</p>", unsafe_allow_html=True)

# --- Sidebar Inputs
with st.sidebar:
    st.markdown("### 🧮 Inputs")
    language = st.radio("Language", ("en", "hi"), index=0)
    N = st.slider("🌿 Nitrogen (N)", 0, 150, 50)
    P = st.slider("🧪 Phosphorus (P)", 0, 150, 50)
    K = st.slider("⚗️ Potassium (K)", 0, 150, 50)
    Temp = st.slider("🌡️ Temperature (°C)", 0, 50, 25)
    Hum = st.slider("💧 Humidity (%)", 0, 100, 50)
    ph = st.slider("🧬 pH Level", 0.0, 14.0, 7.0, step=0.1)
    Rain = st.slider("☔ Rainfall (mm)", 0, 300, 100)
    recommend_btn = st.button("🚀 Recommend Crop")

# --- Load Model
@st.cache_resource
def load_model(path="model.pkl"):
    return joblib.load(path)

model = load_model()

# --- Helper to predict
def predict_with_confidence(model, X):
    try:
        probs = model.predict_proba(X)
        idx = np.argmax(probs, axis=1)[0]
        conf = float(probs[0][idx])
        pred = model.classes_[idx]
        return pred, conf
    except Exception:
        return model.predict(X)[0], None

# --- Main Content
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    if recommend_btn:
        input_data = np.array([[N, P, K, Temp, Hum, ph, Rain]])
        crop, conf = predict_with_confidence(model, input_data)

        # Result Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-metric'>🌾 Recommended Crop: {crop}</div>", unsafe_allow_html=True)
        if conf is not None:
            st.markdown(f"<p class='muted'>Confidence: {conf:.2%}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Feature Importance Chart
        try:
            importance = model.feature_importances_ * 100
            features = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]
            df = pd.DataFrame({"Feature": features, "Importance": importance})
            fig = px.bar(df.sort_values("Importance"), x="Importance", y="Feature",
                         orientation="h", color="Importance", color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance not available for this model.")

        # Generate PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("GreenPulse Crop Recommendation Report", styles['Title']), Spacer(1, 12)]
        story.append(Paragraph(f"Recommended Crop: {crop}", styles['Normal']))
        story.append(Spacer(1, 12))
        for label, val in zip(features, input_data[0]):
            story.append(Paragraph(f"{label}: {val}", styles['Normal']))
        doc.build(story)
        buffer.seek(0)

        st.download_button(
            label="📥 Download Report",
            data=buffer,
            file_name="GreenPulse_Report.pdf",
            mime="application/pdf"
        )

with main_col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Recent Inputs")
    if 'analytics' not in st.session_state:
        st.session_state['analytics'] = []
    st.session_state['analytics'].append(
        {"N": N, "P": P, "K": K, "Temp": Temp, "Humidity": Hum, "pH": ph, "Rainfall": Rain}
    )
    df_log = pd.DataFrame(st.session_state['analytics']).tail(10)
    st.dataframe(df_log)
    st.markdown("</div>", unsafe_allow_html=True)

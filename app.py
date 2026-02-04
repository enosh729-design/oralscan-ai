import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import time
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="OralScan AI", layout="wide", page_icon="🦷")

# ---------------- PREMIUM UI ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, #111827, #030712 70%);
    color: #f1f5f9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220, #0f172a);
}
.glass-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border-radius: 18px;
    padding: 1.6rem;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 40px rgba(0,0,0,0.45);
    margin-bottom: 1rem;
}
.risk-high { color:#ff4d4f; }
.risk-mod { color:#facc15; }
.risk-low { color:#22c55e; }
.footer { color:#94a3b8; font-size:0.75rem; text-align:center; margin-top:3rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div style="padding:1.2rem 0 2.2rem 0;">
    <h1 style="font-size:2.8rem;">🧠 OralScan AI</h1>
    <p style="color:#94a3b8;">⚕️ AI-powered clinical intelligence for early oral cancer risk detection</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_data
def load_data_and_model():
    df = pd.read_csv("oral_cancer.csv")
    df.columns = df.columns.str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("Oral_Cancer_Diagnosis", axis=1)
    y = df["Oral_Cancer_Diagnosis"]

    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X, y)

    return model, X.columns, df

model, feature_columns, full_df = load_data_and_model()

# ---------------- PDF FUNCTION ----------------
def create_pdf(age, tumor_size, prob, category, confidence, explanation):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "OralScan AI — Clinical Risk Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Patient Age: {age}")
    c.drawString(50, height - 120, f"Tumor Size: {tumor_size} cm")
    c.drawString(50, height - 160, f"Predicted Risk Level: {category}")
    c.drawString(50, height - 180, f"Estimated Probability: {prob:.0%}")
    c.drawString(50, height - 200, f"AI Confidence: {confidence:.0%}")

    c.drawString(50, height - 240, "Key Contributing Factors:")
    text = c.beginText(70, height - 260)
    text.textLine(explanation)
    c.drawText(text)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧑‍⚕️ Patient Simulation")
    age = st.slider("🎂 Age", 10, 90, 45)
    tumor_size = st.slider("📏 Tumor size (cm)", 0.0, 10.0, 1.0)

    tobacco = st.selectbox("🚬 Tobacco use", ["No", "Yes"])
    alcohol = st.selectbox("🍺 Alcohol", ["No", "Yes"])
    betel = st.selectbox("🌿 Betel quid", ["No", "Yes"])

    lesions = st.selectbox("🩺 Oral lesions", ["No", "Yes"])
    patches = st.selectbox("⚪ Red/white patches", ["No", "Yes"])
    hpv = st.selectbox("🧬 HPV infection", ["No", "Yes"])
    early = st.selectbox("⏱ Early diagnosis", ["No", "Yes"])

    predict_btn = st.button("🚀 Run AI Risk Analysis")

# ---------------- INPUT PREP ----------------
input_dict = {
    "Age": age,
    "Tumor_Size_cm": tumor_size,
    "Tobacco_Use": tobacco == "Yes",
    "Alcohol_Consumption": alcohol == "Yes",
    "Betel_Quid_Use": betel == "Yes",
    "Oral_Lesions": lesions == "Yes",
    "White_or_Red_Patches_in_Mouth": patches == "Yes",
    "HPV_Infection": hpv == "Yes",
    "Early_Diagnosis": early == "Yes",
}
input_df = pd.DataFrame([input_dict])
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.2, 0.8])

# ---------- RISK PANEL ----------
with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Risk Assessment")

    if predict_btn:
        with st.spinner("Analyzing patient risk..."):
            time.sleep(1.2)

        prob = float(model.predict_proba(input_df)[0][1])
        confidence = np.max(model.predict_proba(input_df))

        if prob > 0.7:
            cat, css, emoji = "High Risk", "risk-high", "🚨"
        elif prob > 0.4:
            cat, css, emoji = "Moderate Risk", "risk-mod", "⚠️"
        else:
            cat, css, emoji = "Low Risk", "risk-low", "✅"
            st.balloons()

        st.markdown(f"## {emoji} <span class='{css}'>{cat}</span> — {prob:.0%}", unsafe_allow_html=True)
        st.progress(int(prob * 100))
        st.markdown(f"🧠 **AI Confidence:** {confidence:.0%}")

        importances = pd.Series(model.feature_importances_, index=feature_columns)
        top_features = importances.sort_values(ascending=False).head(3).index.tolist()
        explanation = ", ".join([f.replace("_", " ") for f in top_features])

        st.markdown(f"🔎 **Why this risk?** {explanation}")

        pdf_file = create_pdf(age, tumor_size, prob, cat, confidence, explanation)
        st.download_button("📄 Download Clinical Risk Report", pdf_file, "OralScan_Report.pdf")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FEATURE IMPORTANCE ----------
with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Key AI Drivers")
    feat_imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)[:8]
    fig, ax = plt.subplots()
    feat_imp.sort_values().plot(kind="barh", ax=ax)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    ax.tick_params(colors="white")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- DATA INSIGHTS ----------
st.markdown("## 📈 Population Insights")

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("Diagnosis Distribution")
    st.bar_chart(full_df["Oral_Cancer_Diagnosis"].value_counts())
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if "Cancer_Stage" in full_df.columns:
        st.markdown("Cancer Stage Distribution")
        st.bar_chart(full_df["Cancer_Stage"].value_counts())
    else:
        st.write("Stage data not available.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
⚠️ For research and educational use only. This AI system does not replace professional medical diagnosis.
</div>
""", unsafe_allow_html=True)


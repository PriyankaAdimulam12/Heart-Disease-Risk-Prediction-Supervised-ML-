# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------- CONFIG ----------------
MODEL_PATH = "final_xgb_pipeline.pkl"
st.set_page_config(
    page_title="Ten-Year CHD Risk Predictor", 
    page_icon="❤", 
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    /* Base App Styling */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Main Title */
    .title {
        text-align: center;
        font-size: 3rem;
        color: #000000;
        font-weight: 800;
        padding: 10px 0;
        margin-bottom: 15px;
    }
    
    /* Card Style */
    .card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
    }

    /* Subheaders */
    .stMarkdown h2, .stMarkdown h3 {
        color: #000000;
        font-weight: 700;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 5px;
        margin-bottom: 10px;
    }
    
    /* Labels & Text */
    label, .stRadio label, .stSelectbox label, .stNumberInput label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* Input Fields */
    div[data-testid*="stNumberInput"] input, 
    div[data-testid*="stSelectbox"] div[role="listbox"] > div,
    div[data-testid*="stTextInput"] input {
        color: #000000 !important;
        font-weight: 600;
        background-color: #ffffff !important;
        border: 1px solid #e74c3c !important;
        border-radius: 5px !important;
    }
    
    div[data-testid*="stNumberInput"] div[data-testid="stInputContainer"] > div,
    div[data-testid*="stSelectbox"] div[role="listbox"],
    div[data-testid*="stTextInput"] div[data-testid="stInputContainer"] > div {
        background-color: #ffffff !important;
        border: 1px solid #e74c3c !important;
        border-radius: 5px !important;
    }
    
    div[data-testid*="stSelectbox"] div[data-baseweb="select"] span {
        color: #000000 !important;
    }

    /* Radio Buttons */
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] > p {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Radio button text - more specific selectors */
    .stRadio label p, .stRadio label span, .stRadio label div {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Radio button container text */
    .stRadio > div > label {
        color: #000000 !important;
    }
    
    /* All text within radio button groups */
    .stRadio * {
        color: #000000 !important;
    }
    
    /* Additional radio button text selectors */
    div[data-testid="stRadio"] label {
        color: #000000 !important;
    }
    
    div[data-testid="stRadio"] p {
        color: #000000 !important;
    }
    
    div[data-testid="stRadio"] span {
        color: #000000 !important;
    }
    
    /* Force all radio button text to be black */
    .stRadio label, .stRadio p, .stRadio span, .stRadio div {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        color: #ffffff !important;
        font-weight: bold;
        padding: 12px;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 10px rgba(231, 76, 60, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #c0392b, #a93226);
        box-shadow: 0 6px 15px rgba(231, 76, 60, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e74c3c, #c0392b) !important;
        border-radius: 5px !important;
    }
    
    /* Alert boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        border-left: 4px solid #e74c3c;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 5px;
        color: #000000;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg { 
        background: #ffffff;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 5px;
        overflow: hidden;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: #e74c3c !important;
    }
    
    /* Caption styling */
    .stCaption {
        color: #666666 !important;
        font-style: italic;
        text-align: center;
        margin-top: 15px;
    }
    
    /* Remove extra spacing */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Compact form spacing */
    .stForm {
        padding: 0;
    }
    
    /* Reduce margins */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- UTILS ----------------
@st.cache_resource(show_spinner="Loading machine learning model...")
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Please save the trained pipeline as {path}")
    return joblib.load(path)

def predict_from_inputs(model, df_input: pd.DataFrame):
    """Predicts the risk class and probability from the input DataFrame."""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)[:, 1][0]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(df_input)
            proba = 1 / (1 + np.exp(-decision))[0]
        else:
            proba = float(model.predict(df_input)[0])
        pred = int(proba >= 0.5)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return 0, 0.0
    return pred, float(proba)

# ---------------- APP UI ----------------
st.markdown("<div class='title'>❤ Ten-Year CHD Risk Predictor</div>", unsafe_allow_html=True)

with st.expander("ℹ About this app and Coronary Heart Disease (CHD)"):
    st.markdown("""
This app predicts the *10-year risk of Coronary Heart Disease (CHD)* using a machine learning model.

*Disclaimer:*
⚠ *This is a predictive tool and not a substitute for professional medical diagnosis or advice.* Consult a healthcare provider for any health concerns.
""")

# ---------------- INPUT FORM ----------------
feature_order = [
    "male", "age", "education", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "BMI", "heartRate", "glucose"
]

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Enter Patient Health Metrics")

col1, col2, col3 = st.columns(3)
map_yesno = {"No": 0, "Yes": 1, "Male": 1, "Female": 0}

with col1:
    st.markdown("### Demographics")
    male = st.radio("Sex", ("Female", "Male"), index=0, help="Select patient sex")
    age = st.slider("Age (years)", min_value=18, max_value=100, value=45, help="Patient age in years")
    education = st.selectbox("Education Level (Code)", options=[1, 2, 3, 4], index=1, help="1=Some High School, 2=High School/GED, 3=Some College/Vocational, 4=College/Post-Grad")
    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0.0, max_value=100.0, value=0.0, step=1.0, help="Average number of cigarettes smoked per day")

with col2:
    st.markdown("### Pre-existing Conditions")
    prevalentHyp = st.radio("History of Hypertension?", ("No", "Yes"), index=0, help="Does the patient have pre-existing hypertension?")
    diabetes = st.radio("History of Diabetes?", ("No", "Yes"), index=0, help="Does the patient have pre-existing diabetes?")
    prevalentStroke = st.radio("History of Stroke?", ("No", "Yes"), index=0, help="Has the patient had a stroke previously?")
    BPMeds = st.radio("On BP Medication?", ("No", "Yes"), index=0, help="Is the patient currently taking blood pressure medication?")

with col3:
    st.markdown("### Physiological Measures")
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=200.0, step=1.0, help="Total Cholesterol level")
    glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=500.0, value=85.0, step=1.0, help="Fasting blood glucose level")
    BMI = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=25.0, step=0.1, format="%.1f", help="Body Mass Index")
    heartRate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=180.0, value=70.0, step=1.0, help="Resting heart rate in beats per minute")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DATAFRAME & PREDICTION ----------------
input_data = {
    "male": map_yesno[male],
    "age": float(age),
    "education": int(education),
    "cigsPerDay": float(cigsPerDay),
    "BPMeds": map_yesno[BPMeds],
    "prevalentStroke": map_yesno[prevalentStroke],
    "prevalentHyp": map_yesno[prevalentHyp],
    "diabetes": map_yesno[diabetes],
    "totChol": float(totChol),
    "BMI": float(BMI),
    "heartRate": float(heartRate),
    "glucose": float(glucose),
}

df_input = pd.DataFrame([[input_data[col] for col in feature_order]], columns=feature_order)

with st.expander("🔎 Review Input Data"):
    st.dataframe(df_input.T.rename(columns={0: "Value"}), use_container_width=True)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.markdown("---")
if st.button("🔮 *Calculate 10-Year CHD Risk*"):
    pred_class, prob_pos = predict_from_inputs(model, df_input)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction Result")
    st.write(f"*Predicted 10-Year CHD Risk:* *{prob_pos:.2%}*")
    st.progress(prob_pos)
    if prob_pos >= 0.2:
        st.error("🚨 *Elevated Risk*")
        if prob_pos >= 0.5:
            risk_msg = "⚠ The model predicts a high probability of CHD. *Clinical evaluation is strongly recommended.* Lifestyle changes and medical consultation are crucial."
        else:
            risk_msg = "📈 The model indicates an elevated risk. Consult a doctor to discuss *preventive strategies* and further screening."
    else:
        st.success("✅ *Low Risk*")
        risk_msg = "📉 The model predicts a relatively low risk. Continue to *maintain a healthy lifestyle* and regular check-ups."
    st.info(risk_msg)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed with ❤ using Streamlit & XGBoost | For informational and educational purposes only. Consult a physician for diagnosis.")
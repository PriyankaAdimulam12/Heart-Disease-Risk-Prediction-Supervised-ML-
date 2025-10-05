# 🩺 Framingham Heart Disease Prediction
### Predicting 10-Year Coronary Heart Disease (CHD) Risk Using Machine Learning

---

## 📘 Overview
This project builds a **machine learning model** to predict the **10-year risk of Coronary Heart Disease (CHD)** using data from the **Framingham Heart Study (FHS)** — one of the most influential longitudinal studies in cardiovascular research.  
The model leverages key clinical and demographic features to assess heart disease risk, helping support **preventive healthcare** and **early intervention**.

---

## 📂 Dataset
The dataset originates from the **Framingham Heart Study**, containing patient-level data related to cardiovascular health indicators.

### 🎯 Target Variable
| Column | Description |
|---------|--------------|
| **TenYearCHD** | Binary outcome: 1 if the patient developed CHD within 10 years, 0 otherwise |

### 🔬 Key Features (Risk Factors)
| Category | Columns | Description |
|-----------|----------|-------------|
| **Demographics** | `male`, `age`, `education` | Sex, age at exam, and education level |
| **Behavioral** | `currentSmoker`, `cigsPerDay` | Smoking status and daily cigarette consumption |
| **Medical History** | `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes` | History of BP medication, stroke, hypertension, or diabetes |
| **Clinical Measures** | `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose` | Cholesterol, blood pressure, BMI, heart rate, and fasting glucose |

---

## ⚙️ Project Challenges & Solutions

### 🧩 Key Challenges
**Data Imbalance:**  
The dataset typically shows ~85% negative and 15% positive CHD cases.  
✅ *Handled using resampling techniques (e.g., SMOTE) and robust metrics (ROC-AUC, F1-Score).*

**Missing Data:**  
Columns like `totChol`, `glucose`, and `BMI` contain missing values.  
✅ *Handled via imputation strategies.*

**Feature Encoding:**  
Ordinal features like `education` were encoded carefully to preserve order.

---

## 🌐 Streamlit App – Ten-Year CHD Risk Predictor

An interactive **Streamlit web application** is included (`app.py`) to predict CHD risk for an individual based on their health metrics.

---

📦 Framingham-CHD-Prediction
│
├── HDC.ipynb                # Jupyter notebook with full analysis & model training
├── app.py                   # Streamlit app for live CHD risk prediction
├── final_xgb_pipeline.pkl   # Trained XGBoost model (saved pipeline)
├── Project_dataset.csv      # Dataset (if available)
├── README.md                # Project documentation
└── requirements.txt         # Requirements file


### ▶ Run the App
```bash
streamlit run app.py

### 📜 Disclaimer

⚠ This project is for educational and informational purposes only.
The prediction tool should not be used as a substitute for professional medical diagnosis or advice.
Always consult a healthcare provider for medical concerns.
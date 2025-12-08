# 📊 Credit Risk Prediction using Machine Learning

This project predicts whether a loan applicant is **Good Credit Risk** or **Bad Credit Risk** based on demographic and financial attributes. The model is trained on the **German Credit dataset** using machine learning techniques.

The project includes:
- Data preprocessing pipeline
- Random Forest classification model
- Explainability insights (feature importance)
- Interactive ✔️ Streamlit web application for real-time predictions

---

## 🚀 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit app for interactive prediction UI |
| `preprocessing.py` | Data cleaning, encoding & scaling preprocessing script |
| `train_and_insights.py` | Model training, evaluation & feature importance analysis |
| `credit_risk_model.pkl` | Trained Random Forest model |
| `credit_label_encoders.pkl` | Encoders for categorical features |
| `credit_scaler.pkl` | Standard scaler for numerical features |
| `requirements.txt` | List of required Python packages |
| `X_scaled.csv`, `y.csv` | Processed data for model training |

---

## 🧠 Model Details
###

- **Algorithm**: Random Forest Classifier  
- **Target Variable**: `Risk`
  - 0 → Good Credit Risk
  - 1 → Bad Credit Risk
- **Performance Metrics Evaluated**:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - Feature Importance

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <repo-url>
cd credit-risk-prediction
```

### 2️⃣ Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
## ▶️ Run the Streamlit Application
```bash
streamlit run app.py
```
---

## 📌 Input Features
| Category |	Features |
|------|-------------|
| Personal |	Age, Sex, Job |
| Financial |	Duration, Checking Account, Saving Accounts |
| Misc |	Housing, Purpose |

---
## System Flow

Dataset → Preprocessing → Model Training → Save Artifacts → Streamlit Deployment → User Predicts Risk

---
## Output Example

- ✔️ Good Credit Risk
- ❌ Bad Credit Risk

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Alternative models (XGBoost / LightGBM)
- SHAP explainability inside UI
- Cloud deployment (Render / Hugging Face)
- Logging predictions in database

---
## 🏁 Author
Vinayak Tyagi

---

# 💳 Credit Card Fraud Detection System  
### Internship Project → Production-Ready Machine Learning Application

---

##  About This Project

This repository contains my **Credit Card Fraud Detection** project, originally submitted during my **CodeClause Data Science Internship (July 2025)** and later upgraded into a **production-ready, user-friendly machine learning application**.

The goal of this project is not just to build a model, but to show **how data science is used in the real world** from raw data to a system that **non-technical users can actually use**.

---

##  Problem Statement

Credit card fraud is **rare but extremely costly**.  
Financial institutions need systems that can:

- Detect suspicious transactions
- Assign a **risk score** instead of a hard yes/no
- Allow business teams to take decisions like *allow*, *review*, or *block*

This project solves that by predicting a **fraud probability score** for each transaction and presenting it in a **simple, human-understandable dashboard**.

---

## 🗂️ Repository Structure

- CodeClauseInternship_Fraud-Detection/
- │
- ├── internship_submission/
- │ └── credit_card_fraud_detection_basic.ipynb
- │
- |
- ├── production_app/
- │ ├── notebooks/
- |
- │ ├── src/
- │ │ ├── data_loader.py
- │ | ├── evaluate.py
- │ | ├── feature_engineering.py
- │ | ├── preprocessing.py
- │ │ ├── train.py 
- │ │ ├── sampling.py
- │ │ └── predict.py
- | |
- │ ├── models/
- │ │ └── fraud_xgb.pkl
- | |
- │ ├── dashboard/
- │ │ └── dashboard.py
- | |
- | ├── api/
- │ │ └── app.py
- | |
- │ ├── run_training.py
- | |
- │ └── requirements.txt
- │
- └── README.md


---

## 🧪 Internship Submission (July 2025)

The original internship work includes:

- Exploratory Data Analysis (EDA)
- Severe class imbalance handling using **SMOTE**
- Fraud detection using **XGBoost**
- Hyperparameter tuning with **GridSearchCV**
- Model evaluation using **ROC-AUC**
- Model explainability using **SHAP**

📁 Located inside: `internship_submission/`

---

##  Production-Ready Enhancements

After the internship, the project was upgraded to follow **industry best practices**:

###  Modular ML Pipeline
- Clear separation of preprocessing, training, and prediction logic

###  Training–Inference Consistency
- Same preprocessing logic reused during prediction to avoid feature mismatch issues

###  Interactive Streamlit Dashboard
- CSV upload support
- Fraud probability converted into **human-friendly risk levels**
- Clear metrics and visualizations
- Downloadable analysis report
- Designed for **non-technical users**

### Real-World Risk Interpretation

| Fraud Probability | Risk Level | Suggested Action |
|------------------|-----------|------------------|
| < 0.5 | Low Risk | Allow transaction |
| 0.5 – 0.7 | Medium Risk | Manual review |
| > 0.7 | High Risk | Block / Alert |

---

## 🖥️ How to Run This Project on Your System

Follow these steps exactly to run the application locally.

---

###  Clone the Repository

```bash
git clone https://github.com/Prince5104/CodeClauseInternship_Fraud-Detection.git
cd CodeClauseInternship_Fraud-Detection

python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

pip install -r production_app/requirements.txt

python3 production_app/run_training.py

streamlit run production_app/dashboard/dashboard.py
```
###  What the Dashboard Shows

For non-technical users, the dashboard provides:

Fraud probability for each transaction

Clear risk labels:

Low Risk

 Medium Risk

High Risk

Summary metrics

Fraud probability distribution graph

Downloadable CSV report

No coding knowledge is required to use it. 

## If you want to connect with me follow me on:
  Linkedin: [www.linkedin.com/in/prince-raj-tech](https://www.linkedin.com/in/prince-raj-tech/)
  Youtube channel: https://www.youtube.com/@SynapseSpaceData
  

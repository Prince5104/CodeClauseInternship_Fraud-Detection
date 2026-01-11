import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# IMPORTANT: reuse training preprocessing
from src.preprocessing import preprocess

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.write(
    """
    This tool helps identify **potentially fraudulent transactions**.
    Upload transaction data and the system will assess **fraud risk** in a simple,
    human-friendly way.
    """
)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/fraud_xgb.pkl")

model = load_model()

# ------------------------------
# Upload CSV
# ------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload transaction CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully")

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # ------------------------------
    # Predict Button
    # ------------------------------
    if st.button("🚀 Analyze Transactions"):

        with st.spinner("Analyzing transactions..."):

            # ------------------------------
            # Remove target column if present
            # ------------------------------
            if "Class" in df.columns:
                df = df.drop("Class", axis=1)

            # ------------------------------
            # Apply SAME preprocessing as training
            # ------------------------------
            df_processed = preprocess(df.copy())

            # Ensure correct feature order
            expected_features = model.get_booster().feature_names
            df_processed = df_processed[expected_features]

            # ------------------------------
            # Prediction
            # ------------------------------
            probs = model.predict_proba(df_processed)[:, 1]
            df["Fraud_Probability"] = probs

            # ------------------------------
            # Human-friendly risk labels
            # ------------------------------
            def classify_risk(p):
                if p > 0.7:
                    return "High Risk 🚨"
                elif p > 0.5:
                    return "Medium Risk ⚠️"
                else:
                    return "Low Risk ✅"

            df["Risk_Level"] = df["Fraud_Probability"].apply(classify_risk)

        st.success("🎯 Analysis completed")

        # ------------------------------
        # Results Table
        # ------------------------------
        st.subheader("📊 Fraud Risk Results (Sample)")
        st.dataframe(df[["Fraud_Probability", "Risk_Level"]].head(20))

        # ------------------------------
        # Summary Metrics
        # ------------------------------
        st.subheader("📈 Risk Summary")

        high_risk = (df["Fraud_Probability"] > 0.7).sum()
        medium_risk = ((df["Fraud_Probability"] > 0.5) & (df["Fraud_Probability"] <= 0.7)).sum()
        low_risk = (df["Fraud_Probability"] <= 0.5).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 High Risk", high_risk)
        col2.metric("⚠️ Medium Risk", medium_risk)
        col3.metric("✅ Low Risk", low_risk)

        # ------------------------------
        # Graph: Fraud Probability Distribution
        # ------------------------------
        st.subheader("📉 Fraud Probability Distribution")

        fig, ax = plt.subplots()
        ax.hist(df["Fraud_Probability"], bins=30)
        ax.set_xlabel("Fraud Probability")
        ax.set_ylabel("Number of Transactions")
        ax.set_title("Distribution of Fraud Risk Scores")

        st.pyplot(fig)

        # ------------------------------
        # Download Results
        # ------------------------------
        st.subheader("⬇️ Download Results")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Fraud Analysis CSV",
            data=csv,
            file_name="fraud_analysis_results.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please upload a CSV file to begin fraud analysis.")

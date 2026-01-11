import joblib

def load_model(path="models/fraud_xgb.pkl"):
    return joblib.load(path)

def predict(model, data):
    return model.predict_proba(data)[:,1]
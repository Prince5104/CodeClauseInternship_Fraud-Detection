from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/fraud_xgb.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = np.array(request.json["features"]).reshape(1, -1)
    prob = model.predict_proba(data)[0][1]
    return jsonify({"fraud_probability": float(prob)})

if __name__ == "__main__":
    app.run()
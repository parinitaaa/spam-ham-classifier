from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import numpy as np

svm_model = joblib.load("train/svm_model.pkl")
vectorizer = joblib.load("train/vectorizer.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Spam Classifier API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    text = data["message"]
    vec = vectorizer.transform([text])

    
    pred = svm_model.predict(vec)[0]

    score = svm_model.decision_function(vec)[0]

    
    confidence = round(abs(score) / (abs(score) + 1), 2)

    return jsonify({
        "message": text,
        "prediction": "Spam" if pred == 1 else "Ham",
        "confidence score": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)

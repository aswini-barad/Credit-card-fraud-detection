from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    if prediction[0] == 1:
        result = "⚠ Fraud Transaction Detected"
    else:
        result = "✅ Normal Transaction"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
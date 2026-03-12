from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    input_features = []

    # Collect V1–V10 inputs
    for i in range(1, 11):
        value = request.form.get(f"V{i}")

        # Handle empty inputs
        if value is None or value.strip() == "":
            value = 0

        input_features.append(float(value))

    # Collect Amount
    amount = request.form.get("Amount")

    if amount is None or amount.strip() == "":
        amount = 0

    input_features.append(float(amount))

    # Convert to numpy array
    final_features = np.array(input_features).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(final_features)

    # Predict
    prediction = model.predict(scaled_features)

    if prediction[0] == 1:
        result = "⚠️ Fraud Transaction Detected"
    else:
        result = "✅ Normal Transaction"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
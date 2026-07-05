from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sc = joblib.load(os.path.join(BASE_DIR, "models", "sc.sav"))
model = joblib.load(os.path.join(BASE_DIR, "models", "rf.sav"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    X = np.array([[
        float(request.form["item_weight"]),
        float(request.form["item_fat_content"]),
        float(request.form["item_visibility"]),
        float(request.form["item_type"]),
        float(request.form["item_mrp"]),
        float(request.form["outlet_establishment_year"]),
        float(request.form["outlet_size"]),
        float(request.form["outlet_location_type"]),
        float(request.form["outlet_type"])
    ]])

    X = sc.transform(X)
    prediction = model.predict(X)[0]

    return jsonify({"Prediction": float(prediction)})


if __name__ == "__main__":
    app.run(debug=True)

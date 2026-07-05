from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load once
sc = joblib.load(r"C:\Users\asus\Desktop\Major Project\models\sc.sav")
model = joblib.load(r"C:\Users\asus\Desktop\Major Project\models\rf.sav")


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
    app.run(debug=True, port=9457)
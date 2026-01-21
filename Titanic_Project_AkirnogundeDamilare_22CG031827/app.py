import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Import trainer so app can auto-train if model is missing
from model.model_development import train_and_save_model, MODEL_PATH

app = Flask(__name__)

MODEL = None


def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        return

    # If model doesn't exist, try training from CSV
    train_and_save_model()
    MODEL = joblib.load(MODEL_PATH)


@app.before_request
def ensure_model_loaded():
    # Load once
    global MODEL
    if MODEL is None:
        load_model()


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None

    if request.method == "POST":
        pclass = to_int(request.form.get("Pclass"), 3)
        sex = request.form.get("Sex", "male").strip().lower()
        age = to_float(request.form.get("Age"), 22.0)
        fare = to_float(request.form.get("Fare"), 7.25)
        embarked = request.form.get("Embarked", "S").strip().upper()

        # Build input exactly as training expects
        input_df = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "Fare": fare,
            "Embarked": embarked
        }])

        pred = MODEL.predict(input_df)[0]
        prediction_text = "Survived" if int(pred) == 1 else "Did Not Survive"

    return render_template("index.html", prediction=prediction_text)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

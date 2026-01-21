import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "titanic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "titanic_survival_model.pkl")


def train_and_save_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put your Titanic CSV inside /model/titanic.csv"
        )

    df = pd.read_csv(DATA_PATH)

    # Required columns for this project
    required = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Your CSV is missing these required columns: {missing_cols}")

    # Select 5 input features + target
    X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]].copy()
    y = df["Survived"].copy()

    # Define feature types
    numeric_features = ["Pclass", "Age", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # Preprocessing:
    # - numeric: median imputation
    # - categorical: most_frequent imputation + onehot encoding
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")

    # Reload + test prediction (proof of persistence)
    loaded = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "Fare": 7.25,
        "Embarked": "S"
    }])
    pred = loaded.predict(sample)[0]
    print("\nReloaded model prediction on sample input:", "Survived" if pred == 1 else "Did Not Survive")


if __name__ == "__main__":
    train_and_save_model()

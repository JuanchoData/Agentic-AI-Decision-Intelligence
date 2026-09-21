from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "artifacts/models/risk_classifier.joblib"
)


def load_model():
    """
    Load the trained ML model artifact.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run "
            "'python -m src.models.train_model' first."
        )

    return joblib.load(MODEL_PATH)


def predict_risk(
    temperature: float,
    pressure: float,
    throughput: float,
    downtime_minutes: float,
) -> dict:
    """
    Predict whether an observation requires attention.
    """

    artifact = load_model()

    model = artifact["model"]
    features = artifact["features"]

    observation = pd.DataFrame(
        [
            {
                "temperature": temperature,
                "pressure": pressure,
                "throughput": throughput,
                "downtime_minutes": downtime_minutes,
            }
        ]
    )

    observation = observation[features]

    prediction = int(
        model.predict(observation)[0]
    )

    probability = float(
        model.predict_proba(observation)[0, 1]
    )

    label = (
        "attention"
        if prediction == 1
        else "normal"
    )

    return {
        "prediction": label,
        "attention_probability": round(
            probability,
            4,
        ),
    }


if __name__ == "__main__":

    normal_case = predict_risk(
        temperature=70,
        pressure=30,
        throughput=100,
        downtime_minutes=2,
    )

    risk_case = predict_risk(
        temperature=90,
        pressure=42,
        throughput=80,
        downtime_minutes=30,
    )

    print("\n=== NORMAL-LIKE CASE ===")
    print(normal_case)

    print("\n=== HIGH-RISK-LIKE CASE ===")
    print(risk_case)
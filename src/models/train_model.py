from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/operational_data.csv")
MODEL_PATH = Path("artifacts/models/risk_classifier.joblib")
METRICS_PATH = Path("artifacts/metrics/model_metrics.json")


FEATURES = [
    "temperature",
    "pressure",
    "throughput",
    "downtime_minutes",
]

TARGET = "status"


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()

    y = df[TARGET].map(
        {
            "normal": 0,
            "attention": 1,
        }
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, predictions),
            4,
        ),
        "precision": round(
            precision_score(y_test, predictions),
            4,
        ),
        "recall": round(
            recall_score(y_test, predictions),
            4,
        ),
        "f1": round(
            f1_score(y_test, predictions),
            4,
        ),
        "roc_auc": round(
            roc_auc_score(y_test, probabilities),
            4,
        ),
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "features": FEATURES,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\n=== MODEL PERFORMANCE ===")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "normal",
                "attention",
            ],
        )
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
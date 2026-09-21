from pathlib import Path

import numpy as np
import pandas as pd


def generate_operational_data(
    n_rows: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic operational dataset for analytics,
    machine learning, and agentic AI demonstrations.
    """

    rng = np.random.default_rng(random_state)

    dates = pd.date_range(
        start="2025-01-01",
        periods=n_rows,
        freq="h",
    )

    temperature = rng.normal(
        loc=70,
        scale=8,
        size=n_rows,
    )

    pressure = rng.normal(
        loc=30,
        scale=4,
        size=n_rows,
    )

    throughput = rng.normal(
        loc=100,
        scale=15,
        size=n_rows,
    )

    downtime_minutes = rng.exponential(
        scale=8,
        size=n_rows,
    )

    quality_score = (
        95
        - 0.20 * np.abs(temperature - 70)
        - 0.30 * np.abs(pressure - 30)
        - 0.10 * downtime_minutes
        + rng.normal(0, 2, n_rows)
    )

    quality_score = np.clip(
        quality_score,
        0,
        100,
    )

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": temperature.round(2),
            "pressure": pressure.round(2),
            "throughput": throughput.round(2),
            "downtime_minutes": downtime_minutes.round(2),
            "quality_score": quality_score.round(2),
        }
    )

    df["status"] = np.where(
        df["quality_score"] >= 90,
        "normal",
        "attention",
    )

    return df


if __name__ == "__main__":
    output_path = Path(
        "data/raw/operational_data.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = generate_operational_data()

    data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created dataset with {len(data)} rows at: "
        f"{output_path}"
    )
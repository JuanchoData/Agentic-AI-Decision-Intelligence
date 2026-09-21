import json
from pathlib import Path

from src.agents.orchestrator import route_query


OUTPUT_PATH = Path(
    "artifacts/metrics/routing_evaluation.json"
)


TEST_CASES = [
    {
        "query": "Compare normal and attention observations.",
        "expected": "analytics",
    },
    {
        "query": "Show me a summary of the operational data.",
        "expected": "analytics",
    },
    {
        "query": "What variables are correlated with quality?",
        "expected": "analytics",
    },
    {
        "query": "Show the lowest quality observations.",
        "expected": "analytics",
    },
    {
        "query": (
            "What does the documentation recommend "
            "when quality decreases?"
        ),
        "expected": "rag",
    },
    {
        "query": "What does the quality policy say?",
        "expected": "rag",
    },
    {
        "query": "What procedure should be followed for an anomaly?",
        "expected": "rag",
    },
    {
        "query": (
            "Predict risk for temperature=90, "
            "pressure=42, throughput=80, downtime=30."
        ),
        "expected": "ml",
    },
    {
        "query": (
            "What is the probability of risk for "
            "temperature=70 pressure=30 throughput=100 downtime=5?"
        ),
        "expected": "ml",
    },
]


def evaluate_routing():
    correct = 0
    results = []

    for case in TEST_CASES:

        prediction = route_query(
            {
                "query": case["query"]
            }
        )["route"]

        is_correct = (
            prediction == case["expected"]
        )

        correct += int(is_correct)

        results.append(
            {
                "query": case["query"],
                "expected": case["expected"],
                "predicted": prediction,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(TEST_CASES)

    evaluation = {
        "n_examples": len(TEST_CASES),
        "correct": correct,
        "routing_accuracy": round(
            accuracy,
            4,
        ),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=4,
        )

    print("\n=== ROUTING EVALUATION ===")

    print(
        f"Examples: {len(TEST_CASES)}"
    )

    print(
        f"Correct: {correct}"
    )

    print(
        f"Accuracy: {accuracy:.2%}"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    evaluate_routing()
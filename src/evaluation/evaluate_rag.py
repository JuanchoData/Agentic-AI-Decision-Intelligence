import json
from pathlib import Path

from src.rag.rag_tool import retrieve_knowledge


OUTPUT_PATH = Path(
    "artifacts/metrics/rag_evaluation.json"
)


TEST_CASES = [
    {
        "question": (
            "What should happen when a predictive "
            "model detects elevated risk?"
        ),
        "expected_source": "incident_response.md",
    },
    {
        "question": (
            "What should be investigated when "
            "quality performance decreases?"
        ),
        "expected_source": "quality_policy.md",
    },
    {
        "question": (
            "What should operators review when "
            "abnormal conditions occur?"
        ),
        "expected_source": "operations_guidelines.md",
    },
]


def evaluate_rag():
    correct = 0
    results = []

    for case in TEST_CASES:

        retrieved = retrieve_knowledge(
            question=case["question"],
            top_k=3,
        )

        top_source = retrieved[0]["source"]

        is_correct = (
            top_source
            == case["expected_source"]
        )

        correct += int(is_correct)

        results.append(
            {
                "question": case["question"],
                "expected_source": (
                    case["expected_source"]
                ),
                "top_source": top_source,
                "top_score": retrieved[0]["score"],
                "correct": is_correct,
            }
        )

    hit_rate = correct / len(TEST_CASES)

    evaluation = {
        "n_examples": len(TEST_CASES),
        "correct": correct,
        "top1_retrieval_accuracy": round(
            hit_rate,
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

    print("\n=== RAG RETRIEVAL EVALUATION ===")

    print(
        f"Examples: {len(TEST_CASES)}"
    )

    print(
        f"Correct top-1 retrievals: {correct}"
    )

    print(
        f"Top-1 retrieval accuracy: {hit_rate:.2%}"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    evaluate_rag()
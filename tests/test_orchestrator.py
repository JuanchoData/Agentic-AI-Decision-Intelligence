from src.agents.orchestrator import route_query


def test_analytics_routing():
    state = {
        "query": "Compare normal and attention observations."
    }

    result = route_query(state)

    assert result["route"] == "analytics"


def test_rag_routing():
    state = {
        "query": (
            "What does the documentation recommend "
            "when quality decreases?"
        )
    }

    result = route_query(state)

    assert result["route"] == "rag"


def test_ml_routing():
    state = {
        "query": (
            "Predict risk for temperature=90, "
            "pressure=42, throughput=80, downtime=30."
        )
    }

    result = route_query(state)

    assert result["route"] == "ml"
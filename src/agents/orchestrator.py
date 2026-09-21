import json
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.rag.rag_tool import retrieve_knowledge
from src.tools.analytics_tool import (
    compare_status_groups,
    find_low_quality_cases,
    quality_correlations,
    summarize_operations,
)
from src.tools.ml_tool import predict_risk


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):
    query: str
    route: str
    tool_used: str
    result: Any
    response: str


# ============================================================
# ROUTER
# ============================================================

def route_query(state: AgentState) -> dict:
    """
    Determine which tool should handle the user request.
    """

    query = state["query"].lower()

    ml_keywords = [
        "predict",
        "prediction",
        "risk",
        "probability",
        "forecast",
    ]

    analytics_keywords = [
    "average",
    "summary",
    "summar",
    "compare",
    "correlat",
    "relationship",
    "associated",
    "lowest",
    "worst",
    "low quality",
    "data",
    "observations",
    "records",
]

    knowledge_keywords = [
        "policy",
        "guideline",
        "documentation",
        "document",
        "what should",
        "procedure",
        "knowledge",
        "recommend",
    ]

    if any(word in query for word in ml_keywords):
        route = "ml"

    elif any(word in query for word in analytics_keywords):
        route = "analytics"

    elif any(word in query for word in knowledge_keywords):
        route = "rag"

    else:
        route = "rag"

    return {
        "route": route
    }


def select_route(
    state: AgentState,
) -> Literal["analytics", "rag", "ml"]:
    """
    Return the selected graph branch.
    """

    return state["route"]


# ============================================================
# ANALYTICS NODE
# ============================================================

def analytics_node(
    state: AgentState,
) -> dict:
    """
    Select and execute the appropriate analytics function.
    """

    query = state["query"].lower()

    if "compare" in query:
        result = compare_status_groups()
        tool = "compare_status_groups"

    elif (
        "correlation" in query
        or "relationship" in query
        or "associated" in query
    ):
        result = quality_correlations()
        tool = "quality_correlations"

    elif (
        "lowest" in query
        or "worst" in query
        or "low quality" in query
    ):
        result = find_low_quality_cases(
            limit=5
        )
        tool = "find_low_quality_cases"

    else:
        result = summarize_operations()
        tool = "summarize_operations"

    records = result.to_dict(
        orient="records"
    )

    response = json.dumps(
        records,
        indent=2,
        default=str,
    )

    return {
        "tool_used": tool,
        "result": records,
        "response": response,
    }


# ============================================================
# RAG NODE
# ============================================================

def rag_node(
    state: AgentState,
) -> dict:
    """
    Retrieve relevant internal knowledge.
    """

    results = retrieve_knowledge(
        question=state["query"],
        top_k=3,
    )

    response_parts = []

    for result in results:

        response_parts.append(
            f"Source: {result['source']}\n"
            f"Similarity: {result['score']}\n"
            f"{result['text']}"
        )

    response = "\n\n".join(
        response_parts
    )

    return {
        "tool_used": "retrieve_knowledge",
        "result": results,
        "response": response,
    }


# ============================================================
# ML PARAMETER EXTRACTION
# ============================================================

def extract_number(
    query: str,
    names: list[str],
):
    """
    Extract a numeric parameter from a query.

    Examples:
        temperature=80
        temperature: 80
        downtime 20
    """

    for name in names:

        pattern = (
            rf"{name}\s*(?:=|:)?\s*"
            rf"(-?\d+(?:\.\d+)?)"
        )

        match = re.search(
            pattern,
            query,
            re.IGNORECASE,
        )

        if match:
            return float(
                match.group(1)
            )

    return None


# ============================================================
# ML NODE
# ============================================================

def ml_node(
    state: AgentState,
) -> dict:
    """
    Extract input variables and call the predictive model.
    """

    query = state["query"]

    temperature = extract_number(
        query,
        ["temperature", "temp"],
    )

    pressure = extract_number(
        query,
        ["pressure"],
    )

    throughput = extract_number(
        query,
        ["throughput"],
    )

    downtime = extract_number(
        query,
        [
            "downtime_minutes",
            "downtime",
        ],
    )

    parameters = {
        "temperature": temperature,
        "pressure": pressure,
        "throughput": throughput,
        "downtime_minutes": downtime,
    }

    missing = [
        name
        for name, value
        in parameters.items()
        if value is None
    ]

    if missing:

        response = (
            "Prediction requires these variables: "
            "temperature, pressure, throughput, "
            "and downtime. Missing: "
            + ", ".join(missing)
        )

        return {
            "tool_used": "predict_risk",
            "result": {
                "missing_parameters": missing
            },
            "response": response,
        }

    result = predict_risk(
        temperature=temperature,
        pressure=pressure,
        throughput=throughput,
        downtime_minutes=downtime,
    )

    response = json.dumps(
        result,
        indent=2,
    )

    return {
        "tool_used": "predict_risk",
        "result": result,
        "response": response,
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():
    """
    Create and compile the decision-intelligence workflow.
    """

    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "router",
        route_query,
    )

    workflow.add_node(
        "analytics",
        analytics_node,
    )

    workflow.add_node(
        "rag",
        rag_node,
    )

    workflow.add_node(
        "ml",
        ml_node,
    )

    workflow.add_edge(
        START,
        "router",
    )

    workflow.add_conditional_edges(
        "router",
        select_route,
        {
            "analytics": "analytics",
            "rag": "rag",
            "ml": "ml",
        },
    )

    workflow.add_edge(
        "analytics",
        END,
    )

    workflow.add_edge(
        "rag",
        END,
    )

    workflow.add_edge(
        "ml",
        END,
    )

    return workflow.compile()


app = build_graph()


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def run_agent(
    query: str,
) -> dict:
    """
    Execute the decision-intelligence workflow.
    """

    result = app.invoke(
        {
            "query": query
        }
    )

    return {
        "query": query,
        "route": result["route"],
        "tool_used": result.get(
            "tool_used"
        ),
        "result": result.get(
            "result"
        ),
        "response": result.get(
            "response"
        ),
    }


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    questions = [
        (
            "Compare normal and attention "
            "observations."
        ),
        (
            "What does the documentation recommend "
            "when quality performance decreases?"
        ),
        (
            "Predict risk for temperature=90, "
            "pressure=42, throughput=80, "
            "downtime=30."
        ),
    ]

    for question in questions:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"QUESTION: {question}"
        )

        result = run_agent(
            question
        )

        print(
            f"\nROUTE: {result['route']}"
        )

        print(
            f"TOOL: {result['tool_used']}"
        )

        print(
            "\nRESPONSE:"
        )

        print(
            result["response"]
        )
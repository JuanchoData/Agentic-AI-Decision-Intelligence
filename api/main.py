from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.orchestrator import run_agent


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Agentic AI Decision Intelligence API",
    description=(
        "Decision-intelligence platform combining "
        "structured-data analytics, semantic retrieval, "
        "machine-learning inference, and LangGraph orchestration."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="Natural-language request for the system.",
    )


class QueryResponse(BaseModel):
    query: str
    route: str
    tool_used: str | None
    response: str | None


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Agentic AI Decision Intelligence",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_system(
    request: QueryRequest,
):
    """
    Route a natural-language request through
    the LangGraph decision-intelligence workflow.
    """

    try:

        result = run_agent(
            request.query
        )

        return QueryResponse(
            query=result["query"],
            route=result["route"],
            tool_used=result["tool_used"],
            response=result["response"],
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
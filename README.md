# Agentic AI Decision Intelligence

A production-style decision-intelligence platform combining **semantic retrieval (RAG), structured-data analytics, machine learning inference, LangGraph workflow orchestration, FastAPI, Streamlit, Docker, automated testing, and CI/CD**.

The project demonstrates how heterogeneous analytical capabilities can be exposed as specialized tools and coordinated through a single decision workflow.

> **Current execution mode:** deterministic LangGraph tool routing with local semantic retrieval.
> The architecture is designed so an LLM-based router or generation layer can be added without changing the underlying analytics, retrieval, or ML tools.

---

## Architecture

```text
                         User
                           |
                           v
                     Streamlit UI
                           |
                           v
                      FastAPI API
                           |
                           v
                   LangGraph Router
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
     Analytics Tool     RAG Tool        ML Tool
          |                |                |
          v                v                v
       DuckDB       SentenceTransformer  Random Forest
          |                |                |
          v                v                v
  Structured Data    Knowledge Base    Risk Prediction
```

The platform currently supports three specialized workflows:

1. **Structured-data analytics**

   * SQL analytics with DuckDB
   * Group comparisons
   * Summary statistics
   * Correlation analysis
   * Identification of low-performance observations

2. **Semantic knowledge retrieval**

   * Document ingestion
   * Paragraph-aware chunking
   * Sentence Transformer embeddings
   * Semantic similarity search
   * Source-aware retrieval

3. **Machine learning inference**

   * Random Forest classification
   * Probability-based risk prediction
   * Reusable inference interface
   * Reproducible model training

LangGraph coordinates these capabilities through a unified workflow.

---

## Example Queries

### Analytics

```text
Compare normal and attention observations.
```

The router selects:

```text
Route: analytics
Tool: compare_status_groups
```

Example results:

| Status    | Records | Avg. Quality | Avg. Downtime |
| --------- | ------: | -----------: | ------------: |
| Normal    |     808 |        92.80 |          6.73 |
| Attention |     192 |        88.37 |         11.70 |

---

### Knowledge Retrieval

```text
What does the documentation recommend when quality decreases?
```

The router selects:

```text
Route: rag
Tool: retrieve_knowledge
```

The retrieval system searches the internal knowledge base and returns the most semantically relevant passages together with source names and similarity scores.

---

### ML Prediction

```text
Predict risk for temperature=90, pressure=42,
throughput=80, downtime=30.
```

The router selects:

```text
Route: ml
Tool: predict_risk
```

Example:

```json
{
  "prediction": "attention",
  "attention_probability": 0.7674
}
```

---

## Technology Stack

### AI / Machine Learning

* Python
* scikit-learn
* Random Forest
* Sentence Transformers
* Hugging Face
* semantic embeddings
* retrieval-augmented architecture
* LangGraph

### Data

* Pandas
* NumPy
* DuckDB
* SQL

### Application

* FastAPI
* Pydantic
* Streamlit
* REST APIs

### MLOps / Engineering

* pytest
* Docker
* Docker Compose
* GitHub Actions
* CI/CD
* Git
* reproducible model training
* automated evaluation

---

## Project Structure

```text
Agentic-AI-Decision-Intelligence/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── artifacts/
│   ├── metrics/
│   └── models/
│
├── data/
│   ├── knowledge/
│   ├── processed/
│   └── raw/
│
├── src/
│   ├── agents/
│   │   └── orchestrator.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_rag.py
│   │   └── evaluate_routing.py
│   │
│   ├── models/
│   │   └── train_model.py
│   │
│   ├── rag/
│   │   └── rag_tool.py
│   │
│   ├── tools/
│   │   ├── analytics_tool.py
│   │   ├── data_tool.py
│   │   └── ml_tool.py
│   │
│   └── utils/
│       └── generate_sample_data.py
│
├── tests/
├── .github/workflows/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Evaluation

The project includes both conventional software testing and AI-system evaluation.

### Automated Tests

The test suite currently covers:

* DuckDB analytics
* status comparisons
* low-quality case retrieval
* correlation calculations
* FastAPI endpoints
* request validation
* LangGraph routing

Current result:

```text
10 tests passed
```

### Routing Evaluation

A benchmark set of natural-language requests evaluates whether the workflow routes requests to the appropriate capability.

```text
Routing benchmark examples: 9
Routing accuracy: 100%
```

### RAG Retrieval Evaluation

A separate benchmark verifies whether the expected source document is returned as the top semantic retrieval result.

```text
RAG benchmark examples: 3
Top-1 retrieval accuracy: 100%
```

These benchmarks are intentionally small demonstration datasets and should not be interpreted as production-scale performance estimates.

---

## API

Start the FastAPI backend:

```bash
python -m uvicorn api.main:app --reload
```

API documentation is available locally at:

```text
http://127.0.0.1:8000/docs
```

### Example Request

```json
{
  "query": "Compare normal and attention observations."
}
```

### Example Response

```json
{
  "query": "Compare normal and attention observations.",
  "route": "analytics",
  "tool_used": "compare_status_groups",
  "response": "..."
}
```

---

## Streamlit Application

Start the dashboard:

```bash
python -m streamlit run app/streamlit_app.py
```

The application provides a user interface for:

* analytical requests
* document retrieval
* ML predictions
* workflow route inspection
* tool-selection visibility

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/JuanchoData/Agentic-AI-Decision-Intelligence.git
cd Agentic-AI-Decision-Intelligence
```

### 2. Create a Python environment

```bash
python -m venv agenenvt
```

Windows:

```bash
.\agenenvt\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the demonstration dataset

```bash
python -m src.utils.generate_sample_data
```

### 5. Train the model

```bash
python -m src.models.train_model
```

### 6. Run tests

```bash
python -m pytest -v
```

### 7. Evaluate routing

```bash
python -m src.evaluation.evaluate_routing
```

### 8. Evaluate semantic retrieval

```bash
python -m src.evaluation.evaluate_rag
```

---

## Docker

The repository includes:

* `Dockerfile`
* `docker-compose.yml`
* `.dockerignore`

The container architecture separates the FastAPI backend and Streamlit frontend while allowing both services to use the same application code.

```text
Streamlit container
       |
       | HTTP
       v
FastAPI container
       |
       v
LangGraph workflow
```

---

## CI/CD

GitHub Actions automatically performs:

```text
Checkout repository
        ↓
Configure Python
        ↓
Install dependencies
        ↓
Generate synthetic data
        ↓
Train ML model
        ↓
Run automated tests
        ↓
Evaluate routing
```

This verifies that the application can be recreated from source in a clean environment rather than depending on local artifacts.

---

## Design Decisions

### Why DuckDB?

DuckDB provides an embedded analytical SQL engine that can query CSV and tabular data without requiring a separate database server.

### Why LangGraph?

LangGraph provides an explicit workflow architecture for routing requests between specialized analytical capabilities while maintaining a clear state and execution graph.

### Why Sentence Transformers?

Sentence Transformers convert documents and user questions into vector embeddings that enable semantic retrieval rather than simple keyword matching.

### Why deterministic routing?

The current demonstration intentionally uses deterministic routing to keep the application:

* reproducible
* transparent
* zero-cost
* easy to evaluate

The architecture separates routing from tool implementation, allowing an LLM-based router to be introduced later without redesigning the analytics, ML, API, or retrieval layers.

---

## Future Extensions

Potential extensions include:

* LLM-based dynamic tool selection
* generative RAG responses
* MCP tool servers
* persistent vector databases
* multi-agent workflows
* human feedback loops
* prompt and agent evaluation
* MLflow experiment tracking
* cloud deployment
* authentication and authorization
* persistent structured databases
* observability and tracing
* domain-specific knowledge bases

---

## Key Engineering Concepts Demonstrated

This project demonstrates practical experience with:

**Agentic architecture • LangGraph • RAG • embeddings • semantic search • machine learning • SQL analytics • DuckDB • REST APIs • FastAPI • Streamlit • Docker • automated testing • AI-system evaluation • GitHub Actions • CI/CD**

---

## Author

**Juan Camilo Rojas**

GitHub: [JuanchoData](https://github.com/JuanchoData)

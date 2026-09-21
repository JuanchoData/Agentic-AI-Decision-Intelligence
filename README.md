\# Agentic AI Decision Intelligence



A production-style decision-intelligence platform combining \*\*semantic retrieval (RAG), structured-data analytics, machine learning inference, LangGraph workflow orchestration, FastAPI, Streamlit, Docker, automated testing, and CI/CD\*\*.



The project demonstrates how heterogeneous analytical capabilities can be exposed as specialized tools and coordinated through a single decision workflow.



> \*\*Current execution mode:\*\* deterministic LangGraph tool routing with local semantic retrieval.

> The architecture is designed so an LLM-based router or generation layer can be added without changing the underlying analytics, retrieval, or ML tools.



\---



\## Architecture



```text

&#x20;                        User

&#x20;                          |

&#x20;                          v

&#x20;                    Streamlit UI

&#x20;                          |

&#x20;                          v

&#x20;                     FastAPI API

&#x20;                          |

&#x20;                          v

&#x20;                  LangGraph Router

&#x20;                          |

&#x20;         +----------------+----------------+

&#x20;         |                |                |

&#x20;         v                v                v

&#x20;    Analytics Tool     RAG Tool        ML Tool

&#x20;         |                |                |

&#x20;         v                v                v

&#x20;      DuckDB       SentenceTransformer  Random Forest

&#x20;         |                |                |

&#x20;         v                v                v

&#x20; Structured Data    Knowledge Base    Risk Prediction

```



The platform currently supports three specialized workflows:



1\. \*\*Structured-data analytics\*\*



&#x20;  \* SQL analytics with DuckDB

&#x20;  \* Group comparisons

&#x20;  \* Summary statistics

&#x20;  \* Correlation analysis

&#x20;  \* Identification of low-performance observations



2\. \*\*Semantic knowledge retrieval\*\*



&#x20;  \* Document ingestion

&#x20;  \* Paragraph-aware chunking

&#x20;  \* Sentence Transformer embeddings

&#x20;  \* Semantic similarity search

&#x20;  \* Source-aware retrieval



3\. \*\*Machine learning inference\*\*



&#x20;  \* Random Forest classification

&#x20;  \* Probability-based risk prediction

&#x20;  \* Reusable inference interface

&#x20;  \* Reproducible model training



LangGraph coordinates these capabilities through a unified workflow.



\---



\## Example Queries



\### Analytics



```text

Compare normal and attention observations.

```



The router selects:



```text

Route: analytics

Tool: compare\_status\_groups

```



Example results:



| Status    | Records | Avg. Quality | Avg. Downtime |

| --------- | ------: | -----------: | ------------: |

| Normal    |     808 |        92.80 |          6.73 |

| Attention |     192 |        88.37 |         11.70 |



\---



\### Knowledge Retrieval



```text

What does the documentation recommend when quality decreases?

```



The router selects:



```text

Route: rag

Tool: retrieve\_knowledge

```



The retrieval system searches the internal knowledge base and returns the most semantically relevant passages together with source names and similarity scores.



\---



\### ML Prediction



```text

Predict risk for temperature=90, pressure=42,

throughput=80, downtime=30.

```



The router selects:



```text

Route: ml

Tool: predict\_risk

```



Example:



```json

{

&#x20; "prediction": "attention",

&#x20; "attention\_probability": 0.7674

}

```



\---



\## Technology Stack



\### AI / Machine Learning



\* Python

\* scikit-learn

\* Random Forest

\* Sentence Transformers

\* Hugging Face

\* semantic embeddings

\* retrieval-augmented architecture

\* LangGraph



\### Data



\* Pandas

\* NumPy

\* DuckDB

\* SQL



\### Application



\* FastAPI

\* Pydantic

\* Streamlit

\* REST APIs



\### MLOps / Engineering



\* pytest

\* Docker

\* Docker Compose

\* GitHub Actions

\* CI/CD

\* Git

\* reproducible model training

\* automated evaluation



\---



\## Project Structure



```text

Agentic-AI-Decision-Intelligence/

│

├── api/

│   └── main.py

│

├

```




import json
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/query",
)
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic AI Decision Intelligence",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("Agentic AI Decision Intelligence")

st.write(
    """
    A decision-support platform combining semantic retrieval,
    structured-data analytics, machine-learning inference,
    and LangGraph workflow orchestration.
    """
)


# ============================================================
# EXAMPLE QUERIES
# ============================================================

st.subheader("Example requests")

st.code(
    "Compare normal and attention observations."
)

st.code(
    "What does the documentation recommend when quality decreases?"
)

st.code(
    "Predict risk for temperature=90, pressure=42, "
    "throughput=80, downtime=30."
)


# ============================================================
# USER INPUT
# ============================================================

query = st.text_area(
    "Ask the decision-intelligence system",
    placeholder=(
        "Enter an analytics, knowledge, or prediction request..."
    ),
    height=120,
)


# ============================================================
# QUERY EXECUTION
# ============================================================

if st.button(
    "Run analysis",
    type="primary",
):

    if not query.strip():

        st.warning(
            "Enter a question before running the analysis."
        )

    else:

        try:

            with st.spinner(
                "Running decision workflow..."
            ):

                response = requests.post(
                    API_URL,
                    json={
                        "query": query
                    },
                    timeout=60,
                )

                response.raise_for_status()

                result = response.json()

            # ------------------------------------------------
            # WORKFLOW INFORMATION
            # ------------------------------------------------

            st.success(
                "Workflow completed successfully."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Selected route",
                    result["route"].upper(),
                )

            with col2:
                st.metric(
                    "Tool used",
                    result["tool_used"],
                )

            st.divider()

            # ------------------------------------------------
            # ANALYTICS RESPONSE
            # ------------------------------------------------

            if result["route"] == "analytics":

                st.subheader(
                    "Analytics results"
                )

                try:

                    data = json.loads(
                        result["response"]
                    )

                    df = pd.DataFrame(data)

                    st.dataframe(
                        df,
                        width="stretch",
                    )

                except json.JSONDecodeError:

                    st.write(
                        result["response"]
                    )

            # ------------------------------------------------
            # ML RESPONSE
            # ------------------------------------------------

            elif result["route"] == "ml":

                st.subheader(
                    "Prediction"
                )

                prediction = json.loads(
                    result["response"]
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Predicted status",
                        prediction["prediction"],
                    )

                with col2:

                    probability = (
                        prediction[
                            "attention_probability"
                        ]
                        * 100
                    )

                    st.metric(
                        "Attention probability",
                        f"{probability:.1f}%",
                    )

                st.json(
                    prediction
                )

            # ------------------------------------------------
            # RAG RESPONSE
            # ------------------------------------------------

            elif result["route"] == "rag":

                st.subheader(
                    "Retrieved knowledge"
                )

                st.text(
                    result["response"]
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to the FastAPI backend. "
                "Make sure the API is running on port 8000."
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"API request failed: {error}"
            )

        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )


# ============================================================
# ARCHITECTURE
# ============================================================

st.divider()

st.subheader("System architecture")

st.code(
    """
User Request
     |
     v
FastAPI
     |
     v
LangGraph Router
     |
     +--------+--------+
     |        |        |
     v        v        v
Analytics   RAG       ML
     |        |        |
   DuckDB  Embeddings Random Forest
    """
)
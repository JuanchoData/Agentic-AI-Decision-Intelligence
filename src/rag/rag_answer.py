import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.rag.rag_tool import retrieve_knowledge


load_dotenv()


def format_context(results: list[dict]) -> str:
    """
    Convert retrieved knowledge chunks into
    context that can be supplied to the LLM.
    """

    context_sections = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        section = (
            f"[Source {index}: {result['source']}]\n"
            f"{result['text']}"
        )

        context_sections.append(section)

    return "\n\n".join(context_sections)


def answer_with_rag(
    question: str,
    top_k: int = 3,
) -> dict:
    """
    Retrieve relevant internal knowledge and generate
    a grounded answer using an LLM.
    """

    results = retrieve_knowledge(
        question=question,
        top_k=top_k,
    )

    context = format_context(results)

    model_name = os.getenv(
        "OPENAI_MODEL",
        "gpt-5.6-luna",
    )

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    system_prompt = """
You are a decision-support assistant.

Answer the user's question using ONLY the supplied
knowledge-base context.

Rules:
1. Do not invent information.
2. If the context does not contain enough evidence,
   say that the available documentation is insufficient.
3. Cite supporting sources using their provided
   source names.
4. Distinguish documented guidance from your own
   interpretation.
5. Keep the answer concise and actionable.
"""

    user_prompt = f"""
QUESTION:
{question}

KNOWLEDGE BASE CONTEXT:
{context}
"""

    response = llm.invoke(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                user_prompt,
            ),
        ]
    )

    return {
        "question": question,
        "answer": response.content,
        "sources": [
            {
                "source": result["source"],
                "score": result["score"],
            }
            for result in results
        ],
    }


if __name__ == "__main__":

    question = (
        "What should happen when a predictive "
        "model detects elevated risk?"
    )

    result = answer_with_rag(
        question
    )

    print("\n=== QUESTION ===")
    print(result["question"])

    print("\n=== GROUNDED ANSWER ===")
    print(result["answer"])

    print("\n=== RETRIEVED SOURCES ===")

    for source in result["sources"]:
        print(
            f"{source['source']} "
            f"(similarity={source['score']})"
        )
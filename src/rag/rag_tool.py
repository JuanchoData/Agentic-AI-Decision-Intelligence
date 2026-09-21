from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


KNOWLEDGE_DIR = Path("data/knowledge")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


_embedding_model = None
_cached_chunks = None
_cached_embeddings = None


def get_embedding_model():
    """
    Load the embedding model once and reuse it.
    """

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            MODEL_NAME
        )

    return _embedding_model


def load_documents(
    knowledge_dir: Path = KNOWLEDGE_DIR,
):
    """
    Load Markdown and text documents.
    """

    documents = []

    for path in sorted(knowledge_dir.glob("*")):

        if path.suffix.lower() not in {
            ".md",
            ".txt",
        }:
            continue

        text = path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "source": path.name,
                "text": text,
            }
        )

    if not documents:
        raise ValueError(
            f"No knowledge documents found in {knowledge_dir}"
        )

    return documents


def chunk_text(
    text: str,
    max_chars: int = 700,
):
    """
    Create paragraph-aware chunks without cutting
    sentences at arbitrary character positions.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = []

    current_length = 0

    for paragraph in paragraphs:

        paragraph_length = len(paragraph)

        if (
            current_chunk
            and current_length + paragraph_length > max_chars
        ):
            chunks.append(
                "\n\n".join(current_chunk)
            )

            # Keep the previous paragraph as light overlap.
            current_chunk = [
                current_chunk[-1]
            ]

            current_length = len(
                current_chunk[0]
            )

        current_chunk.append(
            paragraph
        )

        current_length += (
            paragraph_length + 2
        )

    if current_chunk:
        chunks.append(
            "\n\n".join(current_chunk)
        )

    return chunks


def build_knowledge_base(
    force_rebuild: bool = False,
):
    """
    Load documents, create chunks, and calculate embeddings.

    Results are cached in memory so embeddings are not
    recomputed for every question.
    """

    global _cached_chunks
    global _cached_embeddings

    if (
        _cached_chunks is not None
        and _cached_embeddings is not None
        and not force_rebuild
    ):
        return (
            _cached_chunks,
            _cached_embeddings,
        )

    documents = load_documents()

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"]
        )

        for chunk_id, chunk in enumerate(
            document_chunks
        ):

            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

    model = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode_document(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    _cached_chunks = chunks

    _cached_embeddings = np.asarray(
        embeddings
    )

    return (
        _cached_chunks,
        _cached_embeddings,
    )


def retrieve_knowledge(
    question: str,
    top_k: int = 3,
):
    """
    Retrieve the most semantically relevant chunks
    for a question.
    """

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    chunks, document_embeddings = (
        build_knowledge_base()
    )

    model = get_embedding_model()

    query_embedding = model.encode_query(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarities = (
        document_embeddings
        @ query_embedding
    )

    ranked_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in ranked_indices:

        results.append(
            {
                "source": chunks[index]["source"],
                "chunk_id": chunks[index]["chunk_id"],
                "score": round(
                    float(similarities[index]),
                    4,
                ),
                "text": chunks[index]["text"],
            }
        )

    return results


if __name__ == "__main__":

    questions = [
        (
            "What should be investigated when "
            "quality performance decreases?"
        ),
        (
            "What should happen when a predictive "
            "model detects elevated risk?"
        ),
    ]

    for question in questions:

        print(
            f"\nQUESTION: {question}\n"
        )

        results = retrieve_knowledge(
            question,
            top_k=3,
        )

        for number, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"=== RESULT {number} ==="
            )

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Chunk: {result['chunk_id']}"
            )

            print(
                f"Similarity: {result['score']}"
            )

            print(
                result["text"]
            )

            print()
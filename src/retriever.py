import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# How many chunks to retrieve per query
TOP_K = 3


def get_embeddings():
    """Load the same embedding model used during ingestion."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def load_vectorstore():
    """Load the existing ChromaDB vector store from disk."""
    embeddings  = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


def retrieve_relevant_chunks(query: str) -> list:
    """
    Given a user query, return the top-K most relevant chunks.
    Each result is a Document object with .page_content and .metadata.
    """
    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    results     = retriever.invoke(query)
    return results


def format_context(chunks: list) -> str:
    """Combine retrieved chunks into a single context string for the LLM."""
    return "\n\n---\n\n".join([chunk.page_content for chunk in chunks])


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "What is this document about?"
    print(f"Query: {test_query}\n")

    chunks = retrieve_relevant_chunks(test_query)
    print(f"Retrieved {len(chunks)} chunk(s):\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk.page_content[:300])   # print first 300 chars
        print()
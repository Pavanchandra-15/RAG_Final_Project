import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH   = os.path.join(BASE_DIR, "data", "knowledge_base.pdf")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


def load_and_chunk_pdf(pdf_path: str):
    """Load a PDF and split it into overlapping chunks."""
    print(f"[1/3] Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"      → Loaded {len(pages)} page(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"[2/3] Split into {len(chunks)} chunk(s)")
    return chunks


def get_embeddings():
    """Load a free local HuggingFace embedding model."""
    print("      → Loading embedding model (first run downloads ~90MB)...")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def store_embeddings(chunks):
    """Embed each chunk and persist to ChromaDB."""
    print("[3/3] Generating embeddings and storing in ChromaDB...")
    embeddings  = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()
    print(f"      → Done! Vector store saved to: {CHROMA_DIR}")
    return vectorstore


def run_ingestion():
    chunks      = load_and_chunk_pdf(PDF_PATH)
    vectorstore = store_embeddings(chunks)
    return vectorstore


if __name__ == "__main__":
    run_ingestion()
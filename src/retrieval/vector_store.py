import os
import sys
import shutil
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

# Force UTF-8 output encoding for standard output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # Dimension size for all-MiniLM-L6-v2
DEFAULT_COLLECTION_NAME = "college_syllabus_baseline"
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "qdrant_db")

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes local sentence-transformers embedding model.
    Runs locally on CPU, no API key required.
    """
    print(f"[EMBEDDINGS] Loading model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_qdrant_client(db_path: str = DEFAULT_DB_PATH) -> QdrantClient:
    """
    Initializes a local disk-persisted Qdrant client.
    """
    db_full_path = os.path.abspath(db_path)
    print(f"[QDRANT] Connecting to local Qdrant database at: {db_full_path}")
    client = QdrantClient(path=db_full_path)
    return client

def reset_db_directory(db_path: str = DEFAULT_DB_PATH):
    """
    Removes local qdrant_db directory from disk to ensure 100% clean initialization.
    """
    db_full_path = os.path.abspath(db_path)
    if os.path.exists(db_full_path):
        print(f"[CLEANUP] Removing existing local vector database folder at: {db_full_path}")
        shutil.rmtree(db_full_path, ignore_errors=True)

def build_vector_store(
    chunks: list[Document], 
    collection_name: str = DEFAULT_COLLECTION_NAME, 
    db_path: str = DEFAULT_DB_PATH
) -> QdrantVectorStore:
    """
    Creates a fresh Qdrant collection with deterministic integer IDs (0 to N-1).
    Guarantees exactly N unique points stored.
    """
    # Force clean disk wipe of qdrant_db before connecting
    reset_db_directory(db_path)
    
    client = get_qdrant_client(db_path)
    embeddings = get_embedding_model()
    
    print(f"[QDRANT] Creating clean collection '{collection_name}' with vector size {VECTOR_SIZE} (Cosine)...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    # Qdrant accepts valid integers as Point IDs (0, 1, 2, ..., N-1)
    deterministic_ids = list(range(len(chunks)))
    
    print(f"[INDEXING] Embedding and storing {len(chunks)} chunks into Qdrant...")
    vector_store.add_documents(documents=chunks, ids=deterministic_ids)
    
    # Verify exact point count
    actual_count = client.get_collection(collection_name).points_count
    print(f"[SUCCESS] Indexed exactly {actual_count} unique points into Qdrant (Expected: {len(chunks)}).")
    
    return vector_store

def load_existing_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME, 
    db_path: str = DEFAULT_DB_PATH
) -> QdrantVectorStore:
    """
    Loads an existing Qdrant vector store without re-indexing chunks.
    """
    client = get_qdrant_client(db_path)
    embeddings = get_embedding_model()
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

def similarity_search(
    vector_store: QdrantVectorStore, 
    query: str, 
    top_k: int = 4
) -> list[tuple[Document, float]]:
    """
    Performs similarity search against Qdrant collection and returns Top-K (Document, score) pairs.
    """
    print(f"\n[QUERY] Searching for: '{query}' (Top-K={top_k})")
    results = vector_store.similarity_search_with_score(query=query, k=top_k)
    return results

if __name__ == "__main__":
    print("Vector Store service module ready.")

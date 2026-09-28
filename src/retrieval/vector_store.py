import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

# Force UTF-8 output encoding for standard output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # Dimension size for all-MiniLM-L6-v2
DEFAULT_COLLECTION_NAME = "college_syllabus"
DEFAULT_DB_PATH = "./qdrant_db"

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

def build_vector_store(
    chunks: list[Document], 
    collection_name: str = DEFAULT_COLLECTION_NAME, 
    db_path: str = DEFAULT_DB_PATH
) -> QdrantVectorStore:
    """
    Creates/resets a Qdrant collection and indexes document chunks with vectors and payload.
    """
    client = get_qdrant_client(db_path)
    embeddings = get_embedding_model()
    
    # Check if collection exists; re-create to ensure clean baseline build
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"[QDRANT] Collection '{collection_name}' exists. Re-indexing clean baseline...")
        client.delete_collection(collection_name)
    
    print(f"[QDRANT] Creating collection '{collection_name}' with vector size {VECTOR_SIZE} (Cosine)...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    print(f"[INDEXING] Embedding and storing {len(chunks)} chunks into Qdrant...")
    vector_store.add_documents(documents=chunks)
    print(f"[SUCCESS] Successfully indexed all {len(chunks)} chunks into Qdrant!")
    
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

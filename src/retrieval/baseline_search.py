import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import run_pipeline
from src.retrieval.vector_store import build_vector_store, similarity_search

# Test queries specified in Day 2 objectives
TEST_QUERIES = [
    "What subjects are offered in Semester V?",
    "What is the course code for Computer Networks?",
    "How many credits does Machine Learning have?",
    "What topics are covered in Unit III of Computer Networks?",
    "Which semester contains Deep Learning?",
]

def run_baseline_evaluation():
    """
    Builds the baseline vector store and evaluates baseline retrieval across syllabus test queries.
    """
    print("=========================================================")
    print("      DAY 2: BASELINE VECTOR RETRIEVAL EVALUATION        ")
    print("=========================================================\n")
    
    # Path to syllabus
    target_pdf = os.path.join(PROJECT_ROOT, "data", "22MDC Curriculum.pdf")
    
    # 1. Ingestion Pipeline (Day 1)
    raw_docs, chunks = run_pipeline(target_pdf)
    
    # 2. Embedding & Vector Indexing (Day 2)
    vector_store = build_vector_store(
        chunks=chunks, 
        collection_name="college_syllabus_baseline",
        db_path=os.path.join(PROJECT_ROOT, "qdrant_db")
    )
    
    # 3. Query Execution & Results Inspection
    print("\n" + "="*60)
    print("           EXECUTING BASELINE SIMILARITY QUERIES        ")
    print("="*60)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQUERY #{i}: {query}")
        print("-" * 50)
        
        results = similarity_search(vector_store, query=query, top_k=3)
        
        for rank, (doc, score) in enumerate(results, 1):
            page_num = doc.metadata.get("page", "N/A")
            print(f"  Rank #{rank} | Similarity Score: {score:.4f} | Page: {page_num}")
            print(f"  Snippet: {doc.page_content[:200].replace('\n', ' ')}...")
            print("  " + "."*45)

if __name__ == "__main__":
    run_baseline_evaluation()

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 output encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.retrieval.vector_store import get_qdrant_client, get_embedding_model, DEFAULT_COLLECTION_NAME, DEFAULT_DB_PATH

def run_diagnostics(collection_name: str = DEFAULT_COLLECTION_NAME, db_path: str = DEFAULT_DB_PATH):
    print("=" * 70)
    print("      🔍 RUNNING BASELINE RETRIEVAL DIAGNOSTICS (6 CHECKS)")
    print("=" * 70)

    abs_db_path = os.path.abspath(os.path.join(PROJECT_ROOT, db_path.lstrip("./")))
    client = get_qdrant_client(abs_db_path)
    embeddings = get_embedding_model()

    # -------------------------------------------------------------------
    # DIAGNOSTIC 4 — Qdrant Collection Size
    # -------------------------------------------------------------------
    print("\n--- DIAGNOSTIC 4: Qdrant Collection Size Verification ---")
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        point_count = collection_info.points_count
        print(f"Expected chunks/points: 390")
        print(f"Actual points in Qdrant: {point_count}")
        if point_count != 390:
            print(f"⚠️  WARNING: Point count mismatch! (Expected 390, got {point_count})")
            print(f"   -> Multiple index runs detected! ({point_count} / 390 = {point_count // 390} duplicate indexing runs)")
        else:
            print("✅ Point count matches expected 390 chunks.")
    except Exception as e:
        print(f"❌ Error accessing collection '{collection_name}': {e}")
        return

    # -------------------------------------------------------------------
    # DIAGNOSTICS 1, 2, 3 — Retrieval, Point IDs, Score Precision & Chunk Inspection
    # -------------------------------------------------------------------
    test_query = "What subjects are offered in Semester V?"
    print(f"\n--- DIAGNOSTICS 1, 2, 3: Detailed Search Inspection ---")
    print(f"Query: '{test_query}' (Top-K = 4)")

    query_vector = embeddings.embed_query(test_query)
    
    # Query Qdrant using query_points API
    query_response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=4,
        with_payload=True,
        with_vectors=True
    )
    search_results = query_response.points

    print("\n" + "-" * 70)
    seen_ids = set()
    seen_contents = set()
    
    for rank, point in enumerate(search_results, 1):
        point_id = point.id
        score = point.score
        payload = point.payload or {}
        
        page_content = payload.get("page_content", payload.get("text", ""))
        metadata = payload.get("metadata", {})
        page_num = metadata.get("page", payload.get("page", "N/A"))
        
        content_preview = page_content[:180].replace("\n", " ") if page_content else "N/A"
        content_len = len(page_content)
        
        is_duplicate_id = point_id in seen_ids
        seen_ids.add(point_id)
        
        is_duplicate_content = page_content in seen_contents
        seen_contents.add(page_content)

        print(f"Rank #{rank}")
        print(f"  ID               : {point_id} {'(⚠️ DUPLICATE ID!)' if is_duplicate_id else ''}")
        print(f"  Similarity Score : {score:.8f} (8-decimal precision)")
        print(f"  Page Number      : {page_num}")
        print(f"  Chunk Char Length: {content_len}")
        print(f"  Content Preview  : {content_preview}...")
        print(f"  Content Status   : {'⚠️ EXACT DUPLICATE TEXT (from duplicate indexing run)' if is_duplicate_content else 'Distinct chunk text'}")
        print("-" * 70)

    # Classification logic for Diagnostic 3
    print("\n🔍 DIAGNOSTIC 3 CLASSIFICATION:")
    if len(seen_ids) < len(search_results):
        print(" -> RESULT: Case A / Duplicate Point IDs!")
    elif len(seen_contents) < len(search_results):
        print(" -> RESULT: Case A / Duplicate Chunk Ingestion! Different Point IDs contain EXACT DUPLICATE text because the 390 chunks were indexed multiple times into Qdrant!")
    else:
        pages = [p.payload.get("metadata", {}).get("page") for p in search_results]
        if len(set(pages)) == 1:
            print(" -> RESULT: Case B! Top-K results are DIFFERENT chunks extracted from the SAME page.")
        else:
            print(" -> RESULT: Case C! Top-K results are different chunks from different pages.")

    # -------------------------------------------------------------------
    # DIAGNOSTIC 5 — Inspect Stored Vectors
    # -------------------------------------------------------------------
    print("\n--- DIAGNOSTIC 5: Inspect Stored Vectors ---")
    points_to_inspect = search_results[:3]
    for idx, pt in enumerate(points_to_inspect, 1):
        vec = pt.vector
        print(f"Point #{idx} (ID: {pt.id}): Vector Length = {len(vec)}, First 5 floats = {vec[:5]}")
    
    vec1 = points_to_inspect[0].vector
    vec2 = points_to_inspect[1].vector if len(points_to_inspect) > 1 else None
    if vec2 and vec1 == vec2:
        print("⚠️  CONFIRMED: Point 1 and Point 2 have EXACTLY IDENTICAL vectors because they are duplicate copies of the same chunk!")
    else:
        print("✅ Stored vectors are distinct across different points.")

    # -------------------------------------------------------------------
    # DIAGNOSTIC 6 — Verify Chunk-to-Vector Mapping
    # -------------------------------------------------------------------
    print("\n--- DIAGNOSTIC 6: Verify Chunk-to-Vector & Payload Mapping ---")
    sample_pt = search_results[0]
    print(f"Point ID      : {sample_pt.id}")
    print(f"Vector dim    : {len(sample_pt.vector)}")
    print(f"Payload Keys  : {list(sample_pt.payload.keys()) if sample_pt.payload else []}")
    print(f"Payload Metadata: {sample_pt.payload.get('metadata', {})}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_diagnostics("college_syllabus_baseline", "qdrant_db")

import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Force UTF-8 output encoding for standard output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def load_pdf(pdf_path: str) -> list[Document]:
    """
    Loads a PDF file into a list of LangChain Document objects using PyMuPDF.
    Each document corresponds to a single page with metadata (source, page number).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"[LOAD] Loading PDF from: {pdf_path}...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"[SUCCESS] Loaded {len(documents)} pages.")
    return documents

def inspect_documents(documents: list[Document]):
    """
    Inspects document quality, checking for empty pages or potential text extraction issues.
    """
    empty_pages = [doc.metadata.get("page") for doc in documents if not doc.page_content.strip()]
    print(f"\n[INSPECTION SUMMARY]")
    print(f"   - Total Pages: {len(documents)}")
    print(f"   - Empty Pages: {len(empty_pages)} {f'(Pages: {empty_pages})' if empty_pages else ''}")
    
    if documents:
        print("\n--- SAMPLE PAGE 1 METADATA & PREVIEW ---")
        print(f"Metadata: {documents[0].metadata}")
        print(f"Content Preview (first 250 chars):\n{documents[0].page_content[:250]}...")
        print("----------------------------------------\n")

def chunk_documents(
    documents: list[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> list[Document]:
    """
    Splits documents into smaller semantic chunks using RecursiveCharacterTextSplitter.
    Preserves document metadata on every chunk.
    """
    print(f"[CHUNKING] Splitting documents (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[SUCCESS] Created {len(chunks)} text chunks.")
    return chunks

def inspect_chunks(chunks: list[Document], sample_indices: list[int] = [0, 10, 50]):
    """
    Prints sample chunks to visually inspect chunk quality, overlap, and metadata retention.
    """
    print("\n================ INSPECTING CHUNKS ================")
    for idx in sample_indices:
        if idx < len(chunks):
            chunk = chunks[idx]
            print(f"\n--- CHUNK index #{idx} ---")
            print(f"Page Number (Metadata): {chunk.metadata.get('page')}")
            print(f"Chunk Character Length: {len(chunk.page_content)}")
            print(f"Content Preview:\n{chunk.page_content}")
            print("-" * 50)
    print("===================================================\n")

def run_pipeline(pdf_path: str = "data/22MDC Curriculum.pdf"):
    """
    Runs the complete Day 1 ingestion pipeline.
    """
    print("Starting Day 1 RAG Ingestion Pipeline\n")
    
    # Step 1: Load PDF
    raw_docs = load_pdf(pdf_path)
    
    # Step 2: Quality Inspection
    inspect_documents(raw_docs)
    
    # Step 3: Chunking
    chunks = chunk_documents(raw_docs, chunk_size=1000, chunk_overlap=200)
    
    # Step 4: Chunk Inspection & Validation
    inspect_chunks(chunks, sample_indices=[0, 10, 50])
    
    return raw_docs, chunks

if __name__ == "__main__":
    # Resolve root relative path
    project_root = Path(__file__).parent.parent.parent
    target_pdf = os.path.join(project_root, "data", "22MDC Curriculum.pdf")
    run_pipeline(target_pdf)

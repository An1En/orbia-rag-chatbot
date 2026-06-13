"""
STEP 2: DOCUMENT INGESTION PIPELINE
====================================
This script takes all raw documents, splits them into chunks,
creates vector embeddings, and stores them in a vector database.

Think of it like building a library catalog:
- Raw docs = books
- Chunks = individual pages torn from books
- Embeddings = a unique "fingerprint" for each page's content
- Vector DB = the catalog where we store all fingerprints for fast lookup

When a user asks a question later, we convert their question to a fingerprint,
then find the most similar book-pages from our catalog.
"""

import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import (
    TextLoader,      # Loads .txt files
    PyPDFLoader,     # Loads .pdf files
    DirectoryLoader, # Loads all files from a directory
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ─── CONFIGURATION ───────────────────────────────────────────
RAW_DIR = os.path.join("data", "raw")
VECTOR_DIR = os.path.join("vectorstore")

# Step size for splitting: how big each chunk is
CHUNK_SIZE = 1000       # Characters per chunk
CHUNK_OVERLAP = 200     # Overlap between chunks (preserves context across boundaries)

# ─── STEP 1: LOAD DOCUMENTS ──────────────────────────────────
def load_documents():
    """
    Load ALL documents from the data/raw/ directory.
    
    DirectoryLoader automatically picks the right loader based on file extension:
    - .txt files → TextLoader
    - .pdf files → PyPDFLoader
    
    How TextLoader works:
    - Opens a .txt file in UTF-8 encoding
    - Reads the entire content into a Document object
    
    How PyPDFLoader works:
    - Opens a PDF file
    - Extracts text from each page
    - Creates one Document per page
    """
    print("=" * 60)
    print("LOADING DOCUMENTS")
    print("=" * 60)
    
    # Load .txt files - silently skip binary files
    txt_loader = DirectoryLoader(
        RAW_DIR,
        glob="*.txt",           # Only match .txt files
        loader_cls=TextLoader,  # Use TextLoader to load them
        loader_kwargs={"encoding": "utf-8"},  # UTF-8 encoding
        silent_errors=True,     # Skip files that fail to load
    )
    
    # Load .pdf files
    pdf_loader = DirectoryLoader(
        RAW_DIR,
        glob="*.pdf",           # Only match .pdf files
        loader_cls=PyPDFLoader, # Use PyPDFLoader to load them
        silent_errors=True,     # Skip files that fail to load
    )
    
    # Execute both loaders
    txt_docs = txt_loader.load()
    pdf_docs = pdf_loader.load()
    
    # Combine all documents
    all_docs = txt_docs + pdf_docs
    
    print(f"  Loaded {len(txt_docs):3d} text documents")
    print(f"  Loaded {len(pdf_docs):3d} PDF documents")
    print(f"  Total:  {len(all_docs):3d} documents")
    
    return all_docs

# ─── STEP 2: SPLIT INTO CHUNKS ───────────────────────────────
def split_documents(documents):
    """
    Split large documents into smaller chunks for better retrieval.
    
    Why chunk? 
    - LLMs have limited context windows
    - Smaller pieces are easier to search
    - Each chunk should contain ONE coherent topic
    
    RecursiveCharacterTextSplitter is smart about splitting:
    1. First tries to split on paragraph boundaries ("\n\n")
    2. If chunk is still too big, tries line breaks ("\n")
    3. If still too big, tries sentence endings (". ")
    4. If still too big, tries word boundaries (" ")
    5. Last resort: splits by character
    
    The OVERLAP ensures context isn't lost at chunk boundaries.
    E.g., if a sentence spans two chunks, the overlap captures both halves.
    """
    print("\n" + "=" * 60)
    print("SPLITTING DOCUMENTS INTO CHUNKS")
    print("=" * 60)
    
    # Create the splitter with our configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,      # Target size per chunk
        chunk_overlap=CHUNK_OVERLAP, # Overlap between consecutive chunks
        length_function=len,         # Function to measure length (len() in characters)
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority order for splitting
    )
    
    # Perform the splitting
    chunks = text_splitter.split_documents(documents)
    
    print(f"  Created {len(chunks)} chunks")
    print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
    
    return chunks

# ─── STEP 3: CREATE EMBEDDINGS ───────────────────────────────
def create_embeddings():
    """
    Create an embedding model that converts text to vector representations.
    
    What's an embedding?
    - A vector (list of numbers) that captures the "meaning" of text
    - Similar texts have similar vectors (close together in vector space)
    - "dog" and "puppy" → vectors that are close together
    - "dog" and "quantum physics" → vectors that are far apart
    
    We use 'all-MiniLM-L6-v2' because:
    - It's small and fast (only 384 dimensions)
    - Runs 100% locally (no API calls needed)
    - Good accuracy for its size
    - Free and open-source
    """
    print("\n" + "=" * 60)
    print("CREATING EMBEDDING MODEL")
    print("=" * 60)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"  Model: {model_name}")
    print(f"  Dimensions: 384")
    print(f"  Running locally...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use CPU (works everywhere)
        encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine similarity
    )
    
    print("  [OK] Embedding model ready")
    return embeddings

# ─── STEP 4: STORE IN VECTOR DATABASE ────────────────────────
def store_vectors(chunks, embeddings):
    """
    Store document chunks and their embeddings in ChromaDB vector database.
    
    ChromaDB is an open-source vector database that:
    - Stores the original text alongside its vector embedding
    - Supports fast similarity search (find nearest neighbors)
    - Persists to disk (survives restarts)
    - Has a simple API
    
    The flow:
    1. Take each chunk of text
    2. Convert it to a vector using our embedding model
    3. Store both the text AND the vector in ChromaDB
    4. Later, when a user asks a question, we:
       a. Convert the question to a vector
       b. Find the most similar vectors in the database
       c. Return the associated text chunks
    """
    print("\n" + "=" * 60)
    print("STORING IN VECTOR DATABASE")
    print("=" * 60)
    
    # Check if vector store already exists
    if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
        print(f"  Vector store already exists at: {VECTOR_DIR}")
        print(f"  Loading existing store...")
        vectordb = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings,
        )
        print(f"  Existing store has {vectordb._collection.count()} embeddings")
        return vectordb
    
    # Create new vector store from documents
    print(f"  Creating new vector store at: {VECTOR_DIR}")
    print(f"  Processing {len(chunks)} chunks...")
    
    vectordb = Chroma.from_documents(
        documents=chunks,             # Our text chunks
        embedding=embeddings,         # The embedding model
        persist_directory=VECTOR_DIR, # Where to save on disk
    )
    
    # Persist (save) to disk so it survives restarts
    print(f"  [OK] Stored {vectordb._collection.count()} vectors")
    return vectordb

# ─── MAIN PIPELINE ───────────────────────────────────────────
if __name__ == "__main__":
    """
    Run the full ingestion pipeline:
    1. Load documents from disk
    2. Split into chunks
    3. Create embedding model
    4. Store everything in ChromaDB
    
    After this runs once, the vector store is saved to disk.
    Future runs will detect the existing store and skip processing.
    """
    
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = create_embeddings()
    vectordb = store_vectors(chunks, embeddings)
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Documents loaded:  {len(docs)}")
    print(f"  Chunks created:    {len(chunks)}")
    print(f"  Vectors stored:    {vectordb._collection.count()}")
    print(f"  Vector DB location: {os.path.abspath(VECTOR_DIR)}")
    
    # Quick test: what did we store?
    print("\n  Sample chunks from database:")
    results = vectordb.similarity_search("Orbia purpose", k=3)
    for i, doc in enumerate(results):
        print(f"  [{i+1}] {doc.page_content[:100]}...")

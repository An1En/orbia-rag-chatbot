"""
EXPORT EMBEDDINGS FROM CHROMADB TO JSON
========================================
This script extracts all document chunks and their vector embeddings from ChromaDB
and saves them to a single portable JSON file.

Why? So the Vercel deployment can run WITHOUT:
- sentence-transformers (90MB+ model)
- ChromaDB (database server)
- Any heavy ML dependencies

The JSON file contains everything needed for retrieval:
- Each document chunk's text
- Its 384-dimensional embedding vector
- Source metadata

The Vercel serverless function just loads this JSON and computes cosine similarity.
"""

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import get_vectorstore

def export_embeddings(output_path="api/embeddings.json"):
    """Extract all chunks + vectors from ChromaDB and save as JSON."""
    
    print("Loading vector store from ChromaDB...")
    vectordb = get_vectorstore()
    
    # ChromaDB stores data internally. We need to access the raw collection.
    # The _collection attribute gives us the underlying Chroma collection object.
    collection = vectordb._collection
    
    # Get ALL records from the collection
    # get() without arguments returns everything
    print("Fetching all records from ChromaDB...")
    records = collection.get(include=["documents", "metadatas", "embeddings"])
    
    # Structure the data for JSON export
    data = {
        "chunks": [],
        "embedding_dim": None,
    }
    
    for i in range(len(records["ids"])):
        emb = records["embeddings"][i]
        # Convert numpy array to list if needed (JSON can't serialize numpy arrays)
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        chunk = {
            "id": records["ids"][i],
            "text": records["documents"][i],
            "embedding": emb,
            "metadata": records["metadatas"][i] if records["metadatas"] else {},
        }
        data["chunks"].append(chunk)
    
    data["embedding_dim"] = len(records["embeddings"][0]) if records["embeddings"] is not None and len(records["embeddings"]) > 0 else 0
    data["total_chunks"] = len(data["chunks"])
    
    # Write to JSON (compact format to minimize file size)
    print(f"\nWriting {data['total_chunks']} chunks to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"  Embedding dimension: {data['embedding_dim']}")
    print(f"  Total chunks: {data['total_chunks']}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print(f"\n[DONE] Saved to {os.path.abspath(output_path)}")
    
    return data

if __name__ == "__main__":
    export_embeddings()

"""Quick test for the Vercel-compatible RAG engine."""
import warnings, os
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Disable HF tokenizer parallelism to avoid fork issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.vercel_rag import load_embeddings, retrieve, format_context, generate_fallback, add_sources, answer_question

# 1. Test loading
chunks = load_embeddings()
print(f"[OK] Loaded {len(chunks)} chunks from JSON")
print(f"[OK] Embedding dimension: {len(chunks[0]['embedding'])}")

# 2. Test retrieval using first chunk's embedding as query
query_emb = chunks[0]["embedding"]
results = retrieve(query_emb, chunks, top_k=3)

print(f"\n[TEST] Top 3 results (self-similarity should be ~1.0):")
for r in results:
    src = r["chunk"]["metadata"].get("source", "?")
    text_preview = r["chunk"]["text"][:80]
    print(f"  Score: {r['score']:.4f} | {os.path.basename(src)}")
    print(f"  Text: {text_preview}...")

# 3. Test context formatting
context = format_context(results)
print(f"\n[OK] Context formatted: {len(context)} chars")

# 4. Test fallback answer
answer = generate_fallback(context, "What is Orbia purpose")
print(f"\n[TEST] Fallback answer:")
print(f"  {answer[:300]}")

# 5. Test complete pipeline end-to-end (without API key)
full = add_sources(answer, results)
print(f"\n[OK] Full answer with sources: {len(full)} chars")

# 6. Test complete answer_question function
print(f"\n[TEST] answer_question() end-to-end:")
result = answer_question("What is Orbia purpose?")
print(f"  Answer length: {len(result['answer'])}")
print(f"  Sources: {result['sources']}")
print(f"  Fallback: {result['fallback']}")
print(f"  First 200 chars: {result['answer'][:200]}")

print(f"\n{'='*50}")
print(f"[PASS] Vercel RAG engine is ready!")
print(f"{'='*50}")

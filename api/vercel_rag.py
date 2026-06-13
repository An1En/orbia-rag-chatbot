"""
LIGHTWEIGHT RAG ENGINE FOR VERCEL
==================================
Pure-Python RAG pipeline:
- Loads pre-computed embeddings from JSON
- Embeddings via HuggingFace Inference API (all-MiniLM-L6-v2)
- LLM via Groq API (Llama 3)
- NumPy cosine similarity for retrieval
"""

import json
import os
import math
import httpx
from typing import Optional

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.json")

HF_EMBEDDING_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def load_embeddings():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def cosine_similarity(vec_a, vec_b):
    try:
        import numpy as np
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except ImportError:
        dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(y * y for y in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


def retrieve(query_embedding, chunks, top_k=5):
    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "chunk": c} for s, c in scored[:top_k]]


def format_context(results):
    context_parts = []
    for i, r in enumerate(results, 1):
        metadata = r["chunk"].get("metadata", {})
        source = metadata.get("source", "Orbia Document")
        source_name = os.path.basename(source) if source != "Orbia Document" else "Orbia Document"
        text = r["chunk"]["text"][:500]
        context_parts.append(f"[{i}] Source: {source_name}\n{text}")
    return "\n---\n".join(context_parts)


def generate_answer(context, question, api_key):
    """Call Groq Llama 3 to generate an answer from the context."""
    if not api_key:
        return None

    system_prompt = f"""You are an AI assistant specialized in answering questions about Orbia, a global materials company. Your role is to provide accurate, helpful information based ONLY on the context provided below.

CONTEXT:
{context}

RULES:
1. Answer ONLY using the information in the CONTEXT above. If the context doesn't contain enough information to fully answer the question, say "I don't have enough information about that in my knowledge base."
2. Never make up or hallucinate information.
3. Be concise and professional.
4. When relevant, mention which Orbia business group or area the information relates to.
5. If the user asks about something outside Orbia, politely redirect to Orbia-related topics.

USER QUESTION: {question}

ANSWER:"""

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": system_prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] Groq API call failed: {e}")
        return None


def generate_fallback(context, question):
    """Keyword-based fallback when no LLM is available."""
    context_lower = context.lower()
    question_words = [w for w in question.lower().split() if len(w) > 3]

    relevant = []
    for sentence in context.split(". "):
        sentence_lower = sentence.lower()
        matches = sum(1 for word in question_words if word in sentence_lower)
        if matches > 0:
            relevant.append(sentence)

    if relevant:
        answer = "Based on the available information:\n\n"
        answer += "\n".join(f"- {s}" for s in relevant[:5])
        return answer
    else:
        return "I don't have enough information about that in my knowledge base."


def add_sources(answer, results):
    """Append source citations to the answer."""
    result = answer + "\n\n**Sources:**\n"
    seen = set()
    for i, r in enumerate(results, 1):
        source = r["chunk"].get("metadata", {}).get("source", "Unknown")
        source_name = os.path.basename(source)
        if source_name not in seen:
            seen.add(source_name)
            preview = r["chunk"]["text"][:100].replace("\n", " ").strip()
            result += f"[{i}] {source_name}\n    _{preview}..._\n\n"
    return result


def get_query_embedding(text, hf_token=None):
    """
    Convert text to a 384-dim vector using HuggingFace Inference API.
    Uses the same all-MiniLM-L6-v2 model used during local ingestion.
    
    Falls back to environment variable HF_TOKEN if no token is passed.
    """
    token = hf_token or os.environ.get("HF_TOKEN")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                HF_EMBEDDING_URL,
                headers=headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list):
                        return data[0]
                    return data
                return data

            print(f"[WARN] HuggingFace API returned {response.status_code}")
            return None

    except Exception as e:
        print(f"[ERROR] Embedding API error: {e}")
        return None


def answer_question(question, api_key=None, top_k=5):
    """
    Full RAG pipeline: retrieve -> context -> generate -> format.
    
    Args:
        question: The user's question
        api_key: Groq API key (optional, enables LLM answers)
        top_k: Number of chunks to retrieve
    
    Returns:
        dict with 'answer', 'sources', and 'fallback' flag
    """
    # 1. Load chunks
    chunks = load_embeddings()

    # 2. Compute query embedding
    query_emb = get_query_embedding(question)

    if query_emb is None:
        # Embedding API unavailable - fall back to keyword matching
        all_text = "\n".join(c["text"][:300] for c in chunks)
        fallback_answer = generate_fallback(all_text, question)
        return {"answer": fallback_answer, "sources": [], "fallback": True}

    # 3. Retrieve relevant chunks
    results = retrieve(query_emb, chunks, top_k=top_k)

    # 4. Format context
    context = format_context(results)

    # 5. Generate answer via Groq (or fallback)
    answer = generate_answer(context, question, api_key)
    fallback_used = False
    if answer is None:
        answer = generate_fallback(context, question)
        fallback_used = True
        # Return plain answer without sources formatting for fallback
        sources = []
        for r in results:
            source = r["chunk"].get("metadata", {}).get("source", "Unknown")
            sources.append(os.path.basename(source))
        return {
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),
            "fallback": True,
        }

    # 6. Format with sources
    full_answer = add_sources(answer, results)

    # 7. Extract source names
    sources = []
    for r in results:
        source = r["chunk"].get("metadata", {}).get("source", "Unknown")
        sources.append(os.path.basename(source))

    return {
        "answer": full_answer,
        "sources": list(dict.fromkeys(sources)),
        "fallback": False,
    }

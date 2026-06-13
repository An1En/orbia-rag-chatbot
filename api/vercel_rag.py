"""
LIGHTWEIGHT RAG ENGINE FOR VERCEL
==================================
Pure-Python RAG pipeline:
- Loads document chunks from JSON
- Keyword-based retrieval (no external API needed)
- LLM via Groq API (Llama 3)
"""

import json
import os
import httpx

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.json")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def load_embeddings():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


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


def keyword_retrieve(question, chunks, top_k=5):
    question_lower = question.lower()
    question_words = [w for w in question_lower.split() if len(w) > 3]
    scored = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        matches = sum(1 for word in question_words if word in text_lower)
        scored.append((matches, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "chunk": c} for s, c in scored[:top_k]]


def answer_question(question, api_key=None, top_k=5):
    chunks = load_embeddings()
    results = keyword_retrieve(question, chunks, top_k=top_k)
    context = format_context(results)

    answer = generate_answer(context, question, api_key)
    fallback_used = False
    if answer is None:
        answer = generate_fallback(context, question)
        fallback_used = True
        sources = []
        for r in results:
            source = r["chunk"].get("metadata", {}).get("source", "Unknown")
            sources.append(os.path.basename(source))
        return {
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),
            "fallback": True,
        }

    full_answer = add_sources(answer, results)
    sources = []
    for r in results:
        source = r["chunk"].get("metadata", {}).get("source", "Unknown")
        sources.append(os.path.basename(source))

    return {
        "answer": full_answer,
        "sources": list(dict.fromkeys(sources)),
        "fallback": False,
    }

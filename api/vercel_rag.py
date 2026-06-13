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
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        text = r["chunk"]["text"][:1000]
        context_parts.append(f"[{i}] Source: {source_name}\n{text}")
    return "\n---\n".join(context_parts)


def generate_answer(context, question, api_key):
    """Call OpenRouter (free LLM) to generate an answer from the context."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    system_prompt = f"""You are an AI assistant specialized in answering questions about Orbia, a global materials company. Answer the user's question using BOTH the context below AND your general knowledge about Orbia.

CONTEXT (scraped from Orbia's website and reports):
{context}

GUIDELINES:
1. Use the context first, but if it's missing details, supplement with your own knowledge about Orbia.
2. If you're unsure about something, say so — don't make things up.
3. Be thorough and detailed in your answers.
4. If the user asks about something outside Orbia, politely redirect to Orbia-related topics.

USER QUESTION: {question}

ANSWER:"""

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://orbia-rag-chatbot.vercel.app",
                    "X-Title": "Orbia RAG Chatbot",
                },
                json={
                    "model": "meta-llama/llama-3-8b-instruct",
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": system_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] OpenRouter API call failed: {e}")
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


def keyword_retrieve(question, chunks, top_k=10):
    question_lower = question.lower()
    stop_words = {"the","a","an","is","are","was","were","in","on","at","to","for","of","and","or","how","what","who","when","where","why","do","does","did","can","will","would","could","should","may","might","this","that","it","its","with","as","be","by","from","has","have","had","not","no","but","so","if","about"}
    question_words = [w for w in question_lower.split() if w not in stop_words]
    scored = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        matches = sum(1 for word in question_words if word in text_lower)
        scored.append((matches, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, "chunk": c} for s, c in scored[:top_k]]


def answer_question(question, api_key=None, top_k=10):
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

    sources = []
    for r in results:
        source = r["chunk"].get("metadata", {}).get("source", "Unknown")
        sources.append(os.path.basename(source))

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "fallback": False,
    }

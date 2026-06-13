"""
STEP 3: RAG QUERY PIPELINE
===========================
This is the brain of the chatbot. It takes a user question and:
1. Converts the question to a vector (embedding)
2. Finds the most similar document chunks in the vector database
3. Sends those chunks + the question to an LLM
4. Returns a grounded answer with source citations

This is called "Retrieval-Augmented Generation" (RAG) because:
- RETRIEVAL: Find relevant documents from our knowledge base
- AUGMENTED: Add those documents as context
- GENERATION: Have the LLM generate an answer using ONLY that context

The key insight: The LLM never makes up information (hallucinates)
because we force it to answer ONLY from the retrieved documents.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ─── WHERE OUR VECTOR DATABASE LIVES ─────────────────────────
VECTOR_DIR = os.path.join("vectorstore")

# ─── THE SYSTEM PROMPT ───────────────────────────────────────
# This tells the LLM HOW to behave. It's the "personality" of our chatbot.
# The {context} and {question} placeholders are filled in at query time.
SYSTEM_PROMPT = """You are an AI assistant specialized in answering questions about Orbia, a global materials company. Your role is to provide accurate, helpful information based ONLY on the context provided below.

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

# ─── LOAD THE VECTOR STORE ───────────────────────────────────
def get_vectorstore():
    """
    Load the ChromaDB vector store from disk.
    
    This is the database containing all our document chunks and their embeddings.
    We load it once and reuse it for every query.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    vectordb = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings,
    )
    
    return vectordb

# ─── SEARCH FOR RELEVANT DOCUMENTS ───────────────────────────
def retrieve_documents(vectordb, query, k=5):
    """
    Find the k most relevant document chunks for a given query.
    
    How similarity search works:
    1. Convert the user's query to a vector (using the same embedding model)
    2. Measure the "distance" between the query vector and EVERY vector in our DB
    3. Return the k vectors with the smallest distance (most similar)
    4. The associated text chunks are our "retrieved documents"
    
    The distance metric is cosine similarity:
    - Two identical vectors → distance = 0 (exact match)
    - Two completely different vectors → distance = 2 (no relation)
    - We keep the ones with the smallest distances
    
    k controls how many chunks we retrieve. Too few = missing context.
    Too many = too much noise and we exceed the LLM's context window.
    """
    # similarity_search returns a list of Document objects
    # Each Document has: page_content (the text) and metadata (source info)
    docs = vectordb.similarity_search(query, k=k)
    return docs

# ─── FORMAT CONTEXT FROM DOCUMENTS ───────────────────────────
def format_context(docs):
    """
    Convert retrieved documents into a single formatted string for the LLM.
    
    Each document chunk becomes a numbered reference like:
    [1] Source: filename.txt
    First 500 characters of the chunk...
    ---
    [2] Source: report.pdf
    Next chunk...
    
    This format helps the LLM cite sources in its answers.
    """
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        # Get the source filename from metadata (if available)
        source = doc.metadata.get("source", "Unknown")
        source_name = os.path.basename(source) if source != "Unknown" else "Unknown"
        
        # Format this chunk as a reference
        chunk_text = doc.page_content[:500]  # Limit to 500 chars per chunk
        context_parts.append(f"[{i}] Source: {source_name}\n{chunk_text}")
    
    # Join all chunks with a separator
    return "\n---\n".join(context_parts)

# ─── GENERATE ANSWER USING LOCAL LLM (NO API KEY REQUIRED) ───
def generate_answer_local(context, question):
    """
    Generate an answer using a completely local approach.
    
    This is a fallback that works WITHOUT any API key.
    It extracts keywords from the context and matches them to the question.
    
    LIMITATION: This is NOT an LLM. It's a keyword matcher.
    For real LLM answers, use generate_answer_with_groq() below.
    """
    # Simple keyword matching
    context_lower = context.lower()
    question_lower = question.lower()
    
    # Find sentences in context that contain question keywords
    relevant_sentences = []
    for sentence in context.split(". "):
        sentence_lower = sentence.lower()
        # Count how many question words appear in this sentence
        matches = sum(1 for word in question_lower.split() 
                     if len(word) > 3 and word in sentence_lower)
        if matches > 0:
            relevant_sentences.append(sentence)
    
    if relevant_sentences:
        answer = "Based on the available information:\n\n"
        answer += "\n".join(f"- {s}" for s in relevant_sentences[:5])
        return answer
    else:
        return "I don't have enough information about that in my knowledge base."

# ─── GENERATE ANSWER WITH GROQ LLM ───────────────────────────
# Groq provides free access to Llama 3 and other open-source LLMs.
# You need a free API key from https://console.groq.com
#
# If you don't want to use Groq, you can:
# - Use the local fallback above (no API key needed)
# - Replace with OpenAI: pip install langchain-openai
# - Replace with Anthropic: pip install langchain-anthropic

def generate_answer_with_groq(context, question, api_key=None):
    """
    Generate an answer using Groq's Llama 3 LLM.
    
    How this works:
    1. We build a prompt containing: system instructions + context + question
    2. We send this prompt to Groq's API
    3. Groq's Llama 3 model generates a response
    4. We return the response text
    
    The SYSTEM_PROMPT tells the LLM to ONLY use the provided context.
    This prevents hallucination.
    
    Args:
        context: Formatted document chunks (from format_context)
        question: The user's original question
        api_key: Groq API key (or use GROQ_API_KEY env var)
    """
    
    # If no API key provided, fall back to local generation
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("  [INFO] No Groq API key found. Using local fallback.")
        return generate_answer_local(context, question)
    
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Create the LLM instance
        # ChatGroq connects to Groq's API using the Llama 3 model
        llm = ChatGroq(
            model="llama3-8b-8192",  # Free tier model
            temperature=0.2,          # Low temperature = factual, not creative
            api_key=api_key,
        )
        
        # Build the prompt template
        # ChatPromptTemplate lets us define a message template
        # The {context} and {question} placeholders get filled in
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{question}"),
        ])
        
        # Create the chain: prompt → LLM → output parser
        # This is a pipeline that runs sequentially
        chain = prompt | llm | StrOutputParser()
        
        # Execute the chain with our context and question
        # This sends the request to Groq and gets the response
        answer = chain.invoke({
            "context": context,
            "question": question,
        })
        
        return answer
    
    except Exception as e:
        print(f"  [ERROR] Groq API call failed: {e}")
        print("  [INFO] Falling back to local generation.")
        return generate_answer_local(context, question)

# ─── ADD SOURCES TO ANSWER ───────────────────────────────────
def format_answer_with_sources(answer, docs):
    """
    Append source citations to the LLM's answer.
    
    This builds trust by showing WHERE the information came from.
    Each source includes the filename and a preview of the content.
    
    Example:
    Orbia has 5 business groups...
    
    Sources:
    [1] orbia_business_deep_dive.txt
    [2] orbia_corporate_overview_2025.pdf
    """
    # Start with the answer
    result = answer + "\n\n"
    
    # Add sources section
    result += "**Sources:**\n"
    seen_sources = set()
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        source_name = os.path.basename(source)
        
        # Avoid duplicate sources
        if source_name not in seen_sources:
            seen_sources.add(source_name)
            # Show first 100 chars of the chunk as preview
            preview = doc.page_content[:100].replace("\n", " ").strip()
            result += f"[{i}] {source_name}\n"
            result += f"    _{preview}..._\n\n"
    
    return result

# ─── FULL RAG QUERY ──────────────────────────────────────────
def ask(question, vectordb=None, api_key=None, k=5, show_sources=True):
    """
    The main entry point: answer a question using RAG.
    
    Full pipeline:
    1. Retrieve relevant documents from vector store
    2. Format them as context
    3. Generate an answer using LLM (or local fallback)
    4. Optionally append source citations
    
    Args:
        question: The user's question
        vectordb: The vector store (loaded automatically if None)
        api_key: Groq API key
        k: Number of documents to retrieve
        show_sources: Whether to show source citations
    
    Returns:
        The answer string (with or without sources)
    """
    # Load vector store if not provided
    if vectordb is None:
        vectordb = get_vectorstore()
    
    # Step 1: Retrieve relevant documents
    docs = retrieve_documents(vectordb, question, k=k)
    
    # Step 2: Format as context
    context = format_context(docs)
    
    # Step 3: Generate answer
    answer = generate_answer_with_groq(context, question, api_key)
    
    # Step 4: Add sources
    if show_sources:
        result = format_answer_with_sources(answer, docs)
    else:
        result = answer
    
    return result

# ─── QUICK TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Test the RAG pipeline by asking some sample questions.
    """
    print("=" * 60)
    print("TESTING RAG PIPELINE")
    print("=" * 60)
    
    # Load the vector store
    print("\nLoading vector store...")
    vectordb = get_vectorstore()
    
    # Test questions
    test_questions = [
        "What is Orbia's purpose?",
        "What are the five business groups of Orbia?",
        "What does Netafim do?",
    ]
    
    for q in test_questions:
        print(f"\n\n{'─' * 60}")
        print(f"Q: {q}")
        print(f"{'─' * 60}")
        
        answer = ask(q, vectordb=vectordb)
        print(f"A: {answer}")
    
    print("\n\n[DONE] RAG pipeline test complete!")

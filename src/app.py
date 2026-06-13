"""
STEP 4: STREAMLIT CHATBOT UI
=============================
This creates the web interface for our Orbia RAG chatbot.
Users type questions and get answers with source citations.

Streamlit turns Python scripts into web apps automatically.
Each "button press" or "text entry" triggers the script to re-run
from top to bottom (but we cache the vector store so it loads once).
"""

import os
import sys
import streamlit as st

# Add parent directory to path so we can import rag_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import get_vectorstore, retrieve_documents, format_context, generate_answer_with_groq, format_answer_with_sources

# ─── PAGE CONFIGURATION ──────────────────────────────────────
# This must be the FIRST Streamlit command
st.set_page_config(
    page_title="Orbia RAG Chatbot",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CACHING: Load vector store only once ────────────────────
@st.cache_resource
def load_vectorstore():
    """
    @st.cache_resource means this function runs only ONCE.
    Subsequent calls reuse the cached result.
    This prevents reloading the embedding model on every user interaction.
    """
    return get_vectorstore()

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    # Logo and title
    st.markdown(
        """
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #1A1A2E; font-size: 28px;'>🏢 Orbia</h1>
            <p style='color: #666; font-style: italic;'>Advancing life around the world</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    # Groq API Key input (optional)
    st.markdown("### 🔑 LLM Configuration")
    
    api_key = st.text_input(
        "Groq API Key (optional)",
        type="password",
        help="Get a free key at https://console.groq.com. Without it, the chatbot uses keyword matching (limited).",
        placeholder="gsk_...",
    )
    
    st.markdown(
        """
        <div style='background: #FFF3CD; padding: 10px; border-radius: 5px; font-size: 12px;'>
            <strong>💡 No API key?</strong> The chatbot still works using keyword matching. 
            For full LLM-powered answers, get a free Groq API key.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    # Number of documents to retrieve
    k = st.slider(
        "📄 Documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More documents = broader context but slower response. 3-5 is recommended.",
    )
    
    st.divider()
    
    # About section
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        This chatbot uses **Retrieval-Augmented Generation (RAG)** 
        to answer questions about Orbia based on official documents.
        
        **Tech Stack:**
        - LangChain (orchestration)
        - ChromaDB (vector storage)
        - all-MiniLM-L6-v2 (embeddings)
        - Llama 3 via Groq (LLM)
        - Streamlit (UI)
        
        **Data Sources:**
        - Orbia website pages
        - Annual reports (PDF)
        - Business group information
        """
    )
    
    st.divider()
    
    # Quick sample questions
    st.markdown("### 💬 Sample Questions")
    sample_qs = [
        "What is Orbia's purpose?",
        "What are the five business groups?",
        "What does Netafim do?",
        "Tell me about Wavin iConnect",
        "What is Orbia's revenue?",
        "Who is the CEO of Orbia?",
    ]
    for q in sample_qs:
        if st.button(q, use_container_width=True, type="secondary"):
            st.session_state["sample_question"] = q

# ─── MAIN CHAT AREA ──────────────────────────────────────────

# Title
st.markdown(
    """
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <h1 style='color: #1A1A2E;'>🤖 Orbia Knowledge Assistant</h1>
        <p style='color: #666; font-size: 16px;'>
            Ask anything about Orbia — its business groups, purpose, technology, and more.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── CHAT HISTORY ────────────────────────────────────────────
# Initialize chat history in session state (survives re-runs)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm the Orbia Knowledge Assistant. Ask me anything about Orbia's business, purpose, technology, or culture!",
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── HANDLE SAMPLE QUESTION ──────────────────────────────────
if "sample_question" in st.session_state:
    # Get the question and clear it from session state
    question = st.session_state.pop("sample_question")
    
    # Display as user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Add to history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                # Load vector store (cached after first load)
                vectordb = load_vectorstore()
                
                # Step 1: Retrieve relevant documents
                docs = retrieve_documents(vectordb, question, k=k)
                
                # Step 2: Format as context
                context = format_context(docs)
                
                # Step 3: Generate answer
                answer = generate_answer_with_groq(context, question, api_key)
                
                # Step 4: Format with sources
                full_response = format_answer_with_sources(answer, docs)
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ─── CHAT INPUT ──────────────────────────────────────────────
# Text input at the bottom of the page
if prompt := st.chat_input("Ask about Orbia..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                # Load vector store (cached)
                vectordb = load_vectorstore()
                
                # Step 1: Retrieve
                docs = retrieve_documents(vectordb, prompt, k=k)
                
                # Step 2: Format context
                context = format_context(docs)
                
                # Step 3: Generate answer
                answer = generate_answer_with_groq(context, prompt, api_key)
                
                # Step 4: Format with sources
                full_response = format_answer_with_sources(answer, docs)
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ─── FOOTER ──────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #999; font-size: 12px;'>
        <p>Orbia RAG Chatbot — Built with LangChain + ChromaDB + Streamlit</p>
        <p>Data sourced from Orbia's public website and investor materials</p>
    </div>
    """,
    unsafe_allow_html=True,
)

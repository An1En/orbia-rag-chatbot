import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .vercel_rag import answer_question

# ─── PATHS ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ─── CREATE FASTAPI APP ──────────────────────────────────────
app = FastAPI(
    title="Orbia RAG Chatbot API",
    description="Ask questions about Orbia using RAG",
    version="1.0.0",
)

# ─── CORS: Allow any frontend to call this API ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Vercel preview URLs, custom domains)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── REQUEST/RESPONSE MODELS ────────────────────────────────
class QueryRequest(BaseModel):
    """The shape of data the frontend sends to /ask"""
    question: str
    top_k: int = 8               # Number of documents to retrieve

class QueryResponse(BaseModel):
    """The shape of data the API returns"""
    answer: str
    sources: list[str]
    fallback: bool

# ─── API ENDPOINTS ───────────────────────────────────────────

@app.get("/")
def root():
    """Serve the frontend UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {
        "status": "ok",
        "app": "Orbia RAG Chatbot",
        "endpoints": {
            "/": "This message (frontend UI when static/index.html exists)",
            "/ask": "POST a question about Orbia",
            "/health": "Health check",
        },
    }

@app.get("/health")
def health():
    """Simple health check for monitoring."""
    return {"status": "healthy", "chunks_available": True}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Main RAG endpoint: ask a question about Orbia.
    
    Request body:
    {
        "question": "What is Orbia's purpose?",
        "api_key": "gsk_...",  # optional
        "top_k": 5              # optional
    }
    
    Response:
    {
        "answer": "Orbia's purpose is...",
        "sources": ["orbia_overview.txt", ...],
        "fallback": false
    }
    """
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GROQ_API_KEY")
    result = answer_question(
        question=request.question,
        api_key=api_key,
        top_k=request.top_k,
    )
    return QueryResponse(**result)



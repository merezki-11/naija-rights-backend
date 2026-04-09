import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import get_relevant_chunks
from generator import generate_answer

load_dotenv()

app = FastAPI(
    title="Naija Rights API",
    description="AI-powered Nigerian Constitution chatbot",
    version="1.0.0"
)

# CORS — allows the Vercel frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    eli15: bool = False


class Citation(BaseModel):
    chapter: str
    part: str
    section_number: str
    section_title: str
    raw_text: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Naija Rights API"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Step 1: Retrieve relevant chunks — top 8 for wider coverage
        chunks = get_relevant_chunks(request.question, top_k=8)

        # Step 2: Generate answer
        result = generate_answer(
            query=request.question,
            chunks=chunks,
            history=request.history,
            eli15=request.eli15
        )

        return ChatResponse(
            answer=result["answer"],
            citations=[Citation(**c) for c in result["citations"]]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
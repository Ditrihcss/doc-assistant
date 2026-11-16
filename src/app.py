import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.query_rag import load_rag_assets, rag_answer

index_state = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index_state
    print("=== Загрузка RAG-системы ===")
    index_state = load_rag_assets()
    print("=== Система готова к запросам ===")
    yield
    print("=== Остановка сервера ===")


app = FastAPI(title="Doc Assistant API", version="1.0.0", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(..., description="Вопрос к базе документов")
    k: int = Field(default=3, ge=1, le=10, description="Количество фрагментов для поиска")


@app.get("/health")
def health():
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "rag_loaded": index_state is not None}


@app.post("/ask")
def ask(q: Query):
    """Точка входа для вопросов к базе документов (RAG)."""
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    if index_state is None:
        raise HTTPException(status_code=503, detail="RAG-индекс ещё не загружен")

    start_time = time.time()
    answer, sources = rag_answer(q.question, index_state, k=q.k)
    elapsed_time = time.time() - start_time

    return {
        "answer": answer,
        "sources": sources,
        "metrics": {"execution_time_seconds": round(elapsed_time, 2)},
    }

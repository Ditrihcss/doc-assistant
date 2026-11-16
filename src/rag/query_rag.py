import os
import pickle
from pathlib import Path
import faiss
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DB_DIR = BASE_DIR / "data" / "vector_db"
FAISS_INDEX_PATH = DB_DIR / "faiss.index"
DOCS_PICKLE_PATH = DB_DIR / "documents_mapping.pkl"

EMBEDDING_MODEL = "DeepPavlov/rubert-base-cased-sentence"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


def load_rag_assets() -> dict:
    """Загружает FAISS-индекс, чанки и модель эмбеддингов один раз при старте."""
    if not FAISS_INDEX_PATH.exists() or not DOCS_PICKLE_PATH.exists():
        raise FileNotFoundError(
            "Файлы базы FAISS не найдены. Сначала запустите: python -m src.rag.build_index"
        )

    print("Загрузка индекса FAISS и модели эмбеддингов...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(DOCS_PICKLE_PATH, "rb") as f:
        db_data = pickle.load(f)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    return {"index": index, "db_data": db_data, "embedder": embedder}


def search_chunks(question: str, state: dict, k: int = 3) -> list[dict]:
    """Возвращает k наиболее релевантных фрагментов из FAISS."""
    query_vector = state["embedder"].encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    scores, indices = state["index"].search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {
                "text": state["db_data"]["chunks"][idx],
                "distance": float(score),
                "meta": state["db_data"]["metadatas"][idx],
            }
        )
    return results


def rag_answer(question: str, state: dict, k: int = 3):
    """Ищет контекст в FAISS и генерирует ответ через API Ollama."""
    search_results = search_chunks(question, state, k=k)
    context = "\n---\n".join(res["text"] for res in search_results)

    system_prompt = (
        "Ты — строгий ассистент по документам. Отвечай на вопрос, основываясь ТОЛЬКО на предоставленных фрагментах.\n"
        "Если в тексте нет точного ответа, ответь строго фразой: 'В документах не найдено'.\n"
        "Запрещено выдумывать факты или использовать внешние знания."
    )
    user_prompt = f"ФРАГМЕНТЫ ТЕКСТА:\n{context}\n\nВОПРОС: {question}\nОТВЕТ:"

    payload = {
        "model": OLLAMA_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    llm_response = response.json()["choices"][0]["message"]["content"].strip()
    return llm_response, search_results

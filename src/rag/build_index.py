import pickle
from pathlib import Path
import faiss
import nltk
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "clean.csv"
DB_DIR = BASE_DIR / "data" / "vector_db"
FAISS_INDEX_PATH = DB_DIR / "faiss.index"
DOCS_PICKLE_PATH = DB_DIR / "documents_mapping.pkl"

MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence"


def ensure_nltk_punkt():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


def chunk_text_by_sentences(
    text: str,
    tokenizer,
    max_tokens: int = 300,
    overlap_sentences_count: int = 1,
) -> list[str]:
    """Разбивает текст на перекрывающиеся чанки по предложениям в пределах max_tokens."""
    if not isinstance(text, str) or not text.strip():
        return []

    sentences = nltk.sent_tokenize(text, language="russian")
    chunks, current_chunk, current_tokens_count = [], [], 0

    for sentence in sentences:
        sentence_tokens_count = len(tokenizer.encode(sentence, add_special_tokens=False))
        if sentence_tokens_count > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_tokens_count = [], 0
            words = sentence.split()
            sub_chunk, sub_tokens = [], 0
            for word in words:
                word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                if sub_tokens + word_tokens > max_tokens and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk, sub_tokens = [word], word_tokens
                else:
                    sub_chunk.append(word)
                    sub_tokens += word_tokens
            if sub_chunk:
                current_chunk, current_tokens_count = sub_chunk, sub_tokens
            continue

        if current_tokens_count + sentence_tokens_count > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            if overlap_sentences_count > 0 and len(current_chunk) >= overlap_sentences_count:
                current_chunk = current_chunk[-overlap_sentences_count:]
                current_tokens_count = sum(
                    len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk
                )
            else:
                current_chunk, current_tokens_count = [], 0

        current_chunk.append(sentence)
        current_tokens_count += sentence_tokens_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Файл {DATA_PATH} не найден. Сначала выполните prepare.py.")

    ensure_nltk_punkt()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_csv(DATA_PATH)

    print("Загрузка локальной модели эмбеддингов...")
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    all_metadatas = []

    print("Разбиение текстов на чанки...")
    for row in df.itertuples(index=True):
        text = str(row.text)
        topic = str(row.topic) if hasattr(row, "topic") else "unknown"
        doc_id = int(row.Index)

        for chunk in chunk_text_by_sentences(text, tokenizer=tokenizer):
            all_chunks.append(chunk)
            all_metadatas.append({"topic": topic, "doc_id": doc_id})

    print(f"Всего чанков подготовлено: {len(all_chunks)}")

    print("Генерация эмбеддингов...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    # IndexFlatIP + L2-нормализация эквивалентны косинусному сходству
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(DOCS_PICKLE_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadatas": all_metadatas}, f)

    print(f"База FAISS сохранена в {DB_DIR}. Векторов в индексе: {index.ntotal}")


if __name__ == "__main__":
    main()

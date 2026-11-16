from src.gigachat.llm_client import ask_gigachat
from src.rag.query_rag import load_rag_assets, search_chunks


def rag_answer_gigachat(question: str, state: dict, k: int = 3):
    """Ищет контекст в FAISS и формирует ответ через GigaChat API."""
    search_results = search_chunks(question, state, k=k)
    context = "\n---\n".join(res["text"] for res in search_results)

    prompt = (
        "Ты — строгий ассистент по документам. Отвечай на вопрос, основываясь ТОЛЬКО на предоставленных фрагментах.\n"
        "Если в тексте нет точного ответа, ответь строго фразой: 'В документах не найдено'.\n"
        "Запрещено выдумывать факты или использовать внешние знания.\n\n"
        f"ФРАГМЕНТЫ ТЕКСТА:\n{context}\n\n"
        f"ВОПРОС: {question}\n"
        "ОТВЕТ:"
    )

    llm_response = ask_gigachat(prompt, temperature=0.1)
    return llm_response, search_results


if __name__ == "__main__":
    state = load_rag_assets()
    print("RAG-конвейер на FAISS + GigaChat готов.\n")

    for q in [
        "Рецепт приготовления пиццы Маргарита.",
        "Какие новости экономики обсуждаются в документах?",
    ]:
        answer, sources = rag_answer_gigachat(q, state, k=3)
        print(f"Вопрос: {q}")
        print(f"Ответ: {answer}")
        print(f"Найдено источников: {len(sources)}\n")

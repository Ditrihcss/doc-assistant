from datetime import datetime
from src.rag.query_rag import load_rag_assets, search_chunks

_rag_state = None


def _get_rag_state() -> dict:
    """Ленивая однократная загрузка FAISS и модели эмбеддингов."""
    global _rag_state
    if _rag_state is None:
        print("Загрузка векторной базы для агента...")
        _rag_state = load_rag_assets()
    return _rag_state


def tool_search(query: str) -> str:
    """Поиск релевантных фрагментов в базе документов (FAISS)."""
    state = _get_rag_state()
    results = search_chunks(query, state, k=3)
    if not results:
        return "Ничего не найдено."
    return "\n---\n".join(res["text"] for res in results)


def tool_calc(expression: str) -> str:
    """Вычисляет арифметическое выражение, например '2*(3+4)'."""
    # В продакшене вместо eval следует использовать безопасный парсер AST
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Ошибка вычисления: {e}"


def tool_get_date(_: str = "") -> str:
    """Возвращает текущую дату в формате ГГГГ-ММ-ДД."""
    return datetime.now().strftime("%Y-%m-%d")


TOOLS = {
    "search": tool_search,
    "calc": tool_calc,
    "get_date": tool_get_date,
}

TOOLS_DESCRIPTION = """
1. search(query: str) - поиск информации в базе документов.
2. calc(expression: str) - вычисление математических выражений (например, '10 * 8').
3. get_date() - получить текущую дату.
"""

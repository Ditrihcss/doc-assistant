import json
import re
from src.gigachat.llm_client import ask_gigachat

TOPICS = ["Экономика", "Политика", "Спорт", "Технологии", "Культура", "Наука", "Происшествия"]


def classify_news_prompt(text: str) -> str:
    """Формирует структурированный промпт для zero-shot классификации новости."""
    return f"""Ты — ассистент для классификации новостных текстов по тематике.
Выбери ровно ОДНУ тему из закрытого списка: {", ".join(TOPICS)}.

Текст новости: "{text.strip()}"

Ответь СТРОГО валидным JSON без markdown-обёрток (без ```json), со структурой:
{{
    "topic": "одна тема из списка",
    "confidence": 0.0,
    "reason": "краткое обоснование выбора одной фразой"
}}"""


def parse_json_response(raw_response: str) -> dict:
    """Очищает ответ LLM от возможных markdown-блоков и парсит JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip())
    return json.loads(cleaned)


if __name__ == "__main__":
    sample_text = (
        "Центральный банк повысил ключевую ставку до 21%. "
        "Аналитики прогнозируют ослабление рубля и рост инфляции в ближайшем квартале."
    )

    print("1. Формируем промпт для классификации...")
    final_prompt = classify_news_prompt(sample_text)

    print("2. Отправляем промпт в GigaChat...")
    raw_response = ask_gigachat(final_prompt, temperature=0.1)
    print(f"\nСырой ответ модели:\n{raw_response}\n")

    print("3. Проверка валидности полученного JSON...")
    try:
        parsed = parse_json_response(raw_response)
        print("Успех! Распознанный словарь:")
        print(f"Топик: {parsed.get('topic')}")
        print(f"Уверенность: {parsed.get('confidence')}")
        print(f"Обоснование: {parsed.get('reason')}")
    except json.JSONDecodeError:
        print("Ошибка: модель вернула невалидный JSON.")

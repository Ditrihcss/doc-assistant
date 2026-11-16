## Цифровой ассистент

Мини-проект по дисциплине «Современные технологии разработки программных систем искусственного интеллекта».

## Выполнил
* **Студент:** Дитрихов Никита Владимирович
* **Группа:** МП-301

## Структура проекта
- `data/` — датасеты и векторная база FAISS
- `notebooks/` — ноутбук с первичным анализом данных
- `src/classic_model/` — подготовка данных, классификатор на PyTorch, логирование в MLflow
- `src/gigachat/` — работа с GigaChat API и шаблоны промптов
- `src/LoRA/` — эксперименты с LoRA-адаптерами
- `src/rag/` — индексация документов и RAG-поиск
- `src/agent/` — AI-агент с инструментами на Ollama / LangChain
- `src/app.py` — FastAPI сервис

## Запуск
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Перед запуском модулей скопировать `.env.example` в `.env` и заполнить переменные окружения.
Для запуска API локально:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
Или через Docker:
```bash
docker compose up --build
```

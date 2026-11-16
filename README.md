# Интеллектуальный помощник по базе документов (`doc-assistant`)

Учебный мини-проект по дисциплине «Современные технологии разработки программных систем искусственного интеллекта».

## Выполнил
* **Студент:** Дитрихов Никита Владимирович
* **Группа:** МП-301

## Структура проекта

- `data/` — исходные и обработанные данные, `.dvc`-файлы версионирования, векторный индекс FAISS (`data/vector_db/`)
- `notebooks/` — исследовательский ноутбук (`explore.ipynb`)
- `src/classic_model/` — подготовка данных (Т2), PyTorch-классификатор, обучение с чекпоинтами (Т3), трекинг и реестр моделей в MLflow (Т7)
- `src/gigachat/` — клиент GigaChat API и шаблоны промптов (Т4 ч.1)
- `src/LoRA/` — PEFT/LoRA дообучение классификатора и оценка метрик (Т4 ч.2)
- `src/rag/` — построение индекса FAISS и RAG-пайплайн (Т5)
- `src/app.py` — FastAPI инференс-сервис (Т6 ч.1)
- `Dockerfile`, `docker-compose.yml` — контейнеризация сервиса (Т6 ч.2)
- `src/agent/` — AI-агенты с инструментами (ручной ReAct-цикл и LangChain) на базе локальной Ollama (Т8)

## Быстрый старт

1. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Скопируйте `.env.example` в `.env` и заполните ключи (`GIGACHAT_CREDENTIALS`, при необходимости `OLLAMA_URL` и `MLFLOW_TRACKING_URI`).
3. Запуск модулей (из корня репозитория):
   - Подготовка данных: `python -m src.classic_model.prepare`
   - Тест одного шага PyTorch: `python -m src.classic_model.pipline_test`
   - Обучение классификатора: `python -m src.classic_model.train`
   - Обучение с MLflow: `python -m src.classic_model.train_mlflow`
   - LoRA-эксперименты: `python -m src.LoRA.train_lora`
   - Построение индекса FAISS: `python -m src.rag.build_index`
   - FastAPI сервис: `uvicorn src.app:app --host 0.0.0.0 --port 8000`
   - Docker Compose: `docker compose up --build`
   - AI-агент: `python -m src.agent.manual_agent` или `python -m src.agent.langchain_agent`

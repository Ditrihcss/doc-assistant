import os
from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv()


def main():
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise ValueError("Переменная GIGACHAT_CREDENTIALS не найдена в файле .env")

    with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
        models_data = giga.get_models()
        for model in models_data.data:
            print(f"Доступная модель: {model.id_}")


if __name__ == "__main__":
    main()

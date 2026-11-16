from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "raw" / "data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "clean.csv"


def main():
    print("Запуск пайплайна очистки данных...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Исходный файл не найден: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df = df[df["text"].str.len() > 50]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Готово: {len(df)} документов -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

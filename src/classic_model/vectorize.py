from pathlib import Path
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "clean.csv"


def get_prepared_data(max_features: int = 1000):
    """Векторизует весь датасет (используется для быстрого теста пайплайна)."""
    df = pd.read_csv(DATA_PATH)

    vectorizer = TfidfVectorizer(max_features=max_features)
    x_numpy = vectorizer.fit_transform(df["text"]).toarray()

    label_encoder = LabelEncoder()
    y_numpy = label_encoder.fit_transform(df["topic"])

    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.long)

    print(f"Обнаружено классов (тем): {len(label_encoder.classes_)}")
    return x_tensor, y_tensor


def get_split_data(max_features: int = 1000, test_size: float = 0.2, random_state: int = 42):
    """
    Разделяет датасет на train/test ДО обучения TF-IDF и LabelEncoder,
    чтобы исключить утечку данных (data leakage).
    """
    df = pd.read_csv(DATA_PATH)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        df["text"],
        df["topic"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["topic"],
    )

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tr = vectorizer.fit_transform(X_train_raw).toarray()
    X_te = vectorizer.transform(X_test_raw).toarray()

    label_encoder = LabelEncoder()
    y_tr = label_encoder.fit_transform(y_train_raw)
    y_te = label_encoder.transform(y_test_raw)

    print(f"Обнаружено классов (тем): {len(label_encoder.classes_)}")

    return (
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.long),
        label_encoder.classes_,
    )


if __name__ == "__main__":
    get_prepared_data()

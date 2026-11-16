import torch
import torch.nn as nn

from src.classic_model.model import TextClassifier
from src.classic_model.vectorize import get_prepared_data


def main():
    print("Тестовый старт одного шага обучения (sanity check)...")

    x, y = get_prepared_data()

    in_features = x.shape[1]
    num_classes = int(y.max().item()) + 1

    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    print("\nСтруктура нейросети:")
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logits = model(x)
    loss = loss_fn(logits, y)
    print(f"\n[ЗАМЕР] Значение ошибки (Loss) ДО шага обучения: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        new_logits = model(x)
        new_loss = loss_fn(new_logits, y)
    print(f"[ЗАМЕР] Значение ошибки (Loss) ПОСЛЕ шага обучения: {new_loss.item():.4f}")

    if new_loss.item() < loss.item():
        print("Веса скорректированы, ошибка модели пошла вниз.")


if __name__ == "__main__":
    main()

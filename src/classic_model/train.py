from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.classic_model.model import TextClassifier
from src.classic_model.vectorize import get_split_data

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


def main(num_epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
    print("=== Инициализация обучения классификатора ===")

    X_train, y_train, X_test, y_test, classes = get_split_data()

    in_features = X_train.shape[1]
    num_classes = len(classes)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_loss = float("inf")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "best_text_classifier.pth"

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        total_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                logits = model(batch_X)
                loss = loss_fn(logits, batch_y)

                total_test_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                correct_test += (predicted == batch_y).sum().item()
                total_test += batch_y.size(0)

        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = correct_test / total_test

        print(
            f"Эпоха [{epoch + 1:02d}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_test_loss,
                "classes": list(classes),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"--> Чекпоинт сохранён: {checkpoint_path}")

    print("\n=== Обучение успешно завершено ===")


if __name__ == "__main__":
    main()

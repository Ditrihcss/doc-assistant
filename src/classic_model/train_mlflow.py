import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, TensorDataset

from src.classic_model.model import TextClassifier
from src.classic_model.vectorize import get_split_data

load_dotenv()


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / len(loader), correct / total


def main(num_epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, max_features: int = 1000):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("doc-assistant-classifier")

    X_train, y_train, X_test, y_test, classes = get_split_data(max_features=max_features)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = TextClassifier(in_features=X_train.shape[1], num_classes=len(classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(run_name="mlp-tfidf-baseline"):
        mlflow.log_params(
            {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_features": max_features,
                "num_classes": len(classes),
                "optimizer": "Adam",
            }
        )

        best_test_acc = 0.0
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

            train_loss = total_train_loss / len(train_loader)
            train_acc = correct_train / total_train
            test_loss, test_acc = evaluate(model, test_loader, loss_fn)
            best_test_acc = max(best_test_acc, test_acc)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                },
                step=epoch + 1,
            )

            print(
                f"Epoch [{epoch + 1:02d}/{num_epochs}] | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

        mlflow.log_metric("best_test_acc", best_test_acc)
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=X_train[:1].numpy(),
            registered_model_name="DocTextClassifier",
        )
        print("Эксперимент и модель зарегистрированы в MLflow (DocTextClassifier).")


if __name__ == "__main__":
    main()

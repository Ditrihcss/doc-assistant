from pathlib import Path
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.classic_model.model import TextClassifier
from src.classic_model.vectorize import get_split_data

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


def evaluate_model(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module):
    """Вычисляет Loss, Accuracy и weighted F1 на переданном DataLoader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()

            _, predicted = torch.max(logits, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1


def run_lora_experiment(rank_value: int, num_epochs: int = 3) -> float:
    print(f"\n=== Запуск эксперимента LoRA (ранг r = {rank_value}) ===")

    X_train, y_train, X_test, y_test, classes = get_split_data()
    in_features = X_train.shape[1]
    num_classes = len(classes)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    base_model = TextClassifier(in_features=in_features, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()

    _, base_acc, base_f1 = evaluate_model(base_model, test_loader, loss_fn)
    print(f"[Baseline до обучения] Accuracy: {base_acc:.4f}, F1: {base_f1:.4f}")

    # Применяем LoRA к первому линейному слою сети (net.0)
    lora_config = LoraConfig(
        r=rank_value,
        lora_alpha=16,
        target_modules=["net.0"],
        lora_dropout=0.1,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    _, post_acc, post_f1 = evaluate_model(model, test_loader, loss_fn)
    print(f"[После LoRA r={rank_value}] Accuracy: {post_acc:.4f}, F1: {post_f1:.4f}")

    if rank_value == 8:
        output_dir = MODELS_DIR / "lora-adapter"
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        print(f"Веса LoRA-адаптера сохранены в: {output_dir}")

    return post_acc


def main():
    acc_r4 = run_lora_experiment(rank_value=4)
    acc_r16 = run_lora_experiment(rank_value=16)
    acc_r8 = run_lora_experiment(rank_value=8)

    print("\n=== Эксперимент завершён ===")
    print(f"Accuracy: r=4 -> {acc_r4:.4f} | r=8 -> {acc_r8:.4f} | r=16 -> {acc_r16:.4f}")


if __name__ == "__main__":
    main()

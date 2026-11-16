import torch.nn as nn


class TextClassifier(nn.Module):
    """Простой MLP-классификатор текстов поверх TF-IDF векторов."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

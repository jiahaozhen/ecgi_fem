import torch.nn as nn
import torch.nn.functional as F


# --------------------
# MLP Block
# --------------------
class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, D)
        x = self.fc1(x)
        x = F.relu(self.ln1(x))
        x = self.fc2(x)
        x = F.relu(self.ln2(x))
        x = self.dropout(x)
        return x


# --------------------
# MLP Classifier
# --------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=17, hidden_dim=128, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MLPBlock(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(layers)
            ]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (B, T, D)
        x = x.mean(dim=1)  # Pool over time
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.fc(x)


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    from utils.machine_learning_tools import (
        load_dataset,
        split_dataset,
        train_model,
        evaluate_model_dl,
        build_dataloader,
    )

    data_dir = 'machine_learning/data/dataset/d6_standard_dataset'
    X, y = load_dataset(data_dir)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Detect input_dim
    if X_train.ndim == 3:
        input_dim = X_train.shape[2]
    else:
        X_train = X_train[:, None, :]
        X_test = X_test[:, None, :]
        input_dim = X_train.shape[2]

    train_loader = build_dataloader(X_train, y_train)

    model = MLPClassifier(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model_dl(model, X_test, y_test)

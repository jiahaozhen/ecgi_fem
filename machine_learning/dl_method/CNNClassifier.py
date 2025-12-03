import torch.nn as nn
import torch.nn.functional as F


# --------------------
# Residual Block
# --------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel=5):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2)
        self.ln1 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        out = self.conv1(x)
        out = F.relu(self.ln1(out.transpose(1, 2)).transpose(1, 2))
        out = self.conv2(out)
        out = self.ln2(out.transpose(1, 2)).transpose(1, 2)
        return F.relu(out + residual)


# --------------------
# Improved CNN Classifier
# --------------------
class ImprovedCNN(nn.Module):
    def __init__(self, input_dim, n_classes=17):
        super().__init__()

        hidden = 128

        self.proj = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(hidden)
        self.block2 = ResidualBlock(hidden)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, D) → (B, D, T)
        x = F.relu(self.proj(x))
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=2)  # GAP
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

    model = ImprovedCNN(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model_dl(model, X_test, y_test)

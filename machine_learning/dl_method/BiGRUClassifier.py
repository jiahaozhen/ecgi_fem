import torch.nn as nn


# --------------------
# GRU Block
# --------------------
class GRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.gru(x)
        out = self.dropout(out)
        return out


# --------------------
# BiGRU Classifier
# --------------------
class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=17, hidden_dim=128, num_layers=2):
        super().__init__()
        self.bigru = GRUBlock(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (B, T, D)
        out = self.bigru(x)
        out = out.mean(dim=1)  # Global Average Pooling over time
        out = self.dropout(out)
        return self.fc(out)


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

    model = BiGRUClassifier(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model_dl(model, X_test, y_test)

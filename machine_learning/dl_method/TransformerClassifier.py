import torch.nn as nn


# --------------------
# Transformer Block
# --------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


# --------------------
# Transformer Classifier
# --------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=17, hidden=128, layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([TransformerBlock(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (B, T, D)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # Global Average Pooling over time
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

    model = TransformerClassifier(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model_dl(model, X_test, y_test)

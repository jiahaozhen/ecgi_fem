import numpy as np
import h5py
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def load_bsp_pts(path: str = 'forward_inverse_3d/data/geom_ecgsim.mat'):
    geom = h5py.File(path, 'r')
    points = np.array(geom['geom_thorax']['pts'])
    return points


def unwrap_surface(original_pts):
    """
    将环绕身体的 3D 电极点展开到 2D 平面
    返回 2D 坐标 (num_leads, 2)
    """
    X, Y, Z = original_pts[:, 0], original_pts[:, 1], original_pts[:, 2]
    theta = np.arctan2(Y, X)  # 围绕中心旋转角度
    return np.column_stack([theta, Z])  # (num_leads, 2)


def project_bsp_on_surface(bsp_data, original_pts=load_bsp_pts(), length=50, width=50):
    num_timepoints, num_leads = bsp_data.shape

    xy_pts = unwrap_surface(original_pts)  # (num_leads, 2)

    # 构建规则网格
    x_min, x_max = xy_pts[:, 0].min(), xy_pts[:, 0].max()
    y_min, y_max = xy_pts[:, 1].min(), xy_pts[:, 1].max()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, length), np.linspace(y_min, y_max, width)
    )
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # 一次性构建两个插值器
    lin_interp = LinearNDInterpolator(xy_pts, np.zeros(num_leads))
    near_interp = NearestNDInterpolator(xy_pts, np.zeros(num_leads))

    surface_bsp = np.empty((num_timepoints, width, length), dtype=np.float64)

    for t in range(num_timepoints):
        voltages = bsp_data[t, :]

        # 更新插值器数值
        lin_interp.values[:] = voltages.reshape(-1, 1)
        near_interp.values[:] = voltages
        # 线性插值
        grid_z = lin_interp(grid_pts).reshape(width, length)

        # 最近邻插值（完整）
        grid_z_near = near_interp(grid_pts).reshape(width, length)

        # 补洞，不会再 shape mismatch
        nan_mask = np.isnan(grid_z)
        if np.any(nan_mask):
            grid_z[nan_mask] = grid_z_near[nan_mask]

        surface_bsp[t] = grid_z

    return surface_bsp


# ----------------- 机器学习通用工具 -----------------
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_dataset(data_dir):
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    X_list, y_list = [], []
    for fname in file_list:
        data = np.load(os.path.join(data_dir, fname))
        X = data['X'] if 'X' in data else data[list(data.keys())[0]]
        y = data['y'] if 'y' in data else data[list(data.keys())[-1]]
        X_list.append(X)
        y_list.append(y)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_train_test(data_dir, test_size=0.2, random_state=42, test_only=False):
    X, y = load_dataset(data_dir)
    if test_only:
        # 全部数据作为测试集
        return None, X, None, y
    else:
        return split_dataset(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(clf, X_test, y_test):
    y_pred_label = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred_label))
    # print(classification_report(y_test, y_pred_label, digits=4))


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def build_dataloader(X_train, y_train):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader


def train_model(model, train_loader, epochs=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
    return model


def evaluate_model_dl(model, X_test, y_test):
    device = next(model.parameters()).device
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred, digits=4))
    return accuracy_score(y_test, y_pred)

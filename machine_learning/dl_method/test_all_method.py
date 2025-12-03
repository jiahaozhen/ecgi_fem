import time
from machine_learning.dl_method.BiGRUClassifier import BiGRUClassifier
from machine_learning.dl_method.BiLSTMClassifier import BiLSTMClassifier
from machine_learning.dl_method.CNNClassifier import ImprovedCNN
from machine_learning.dl_method.MLPClassifier import MLPClassifier
from machine_learning.dl_method.TCNClassifier import TCNClassifier
from machine_learning.dl_method.TransformerClassifier import TransformerClassifier
from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    build_dataloader,
    train_model,
    evaluate_model_dl,
)

methods = [
    ('BiGRUClassifier', BiGRUClassifier),
    ('BiLSTMClassifier', BiLSTMClassifier),
    ('ImprovedCNN', ImprovedCNN),
    ('MLPClassifier', MLPClassifier),
    ('TCNClassifier', TCNClassifier),
    ('TransformerClassifier', TransformerClassifier),
]
DATA_DIR = 'machine_learning/data/dataset/d6_standard_dataset'


def test_all_classifiers():
    X, y = load_dataset(DATA_DIR)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Detect input_dim
    if X_train.ndim == 3:
        input_dim = X_train.shape[2]
    else:
        X_train = X_train[:, None, :]
        X_test = X_test[:, None, :]
        input_dim = X_train.shape[2]
    results = []

    for name, method in methods:
        train_loader = build_dataloader(X_train, y_train)
        print(f'\n训练 {name}...')
        start_time = time.time()
        try:
            model = method(input_dim)
            model = train_model(model, train_loader, epochs=30, lr=1e-3)
            elapsed = time.time() - start_time
            print(f'{name}: 训练时间 = {elapsed:.4f}s')
            # 评估模型并记录准确度
            print(f'{name} 测试结果:')
            acc = evaluate_model_dl(model, X_test, y_test)
            print('Accuracy:', acc)
            results.append(
                {'method': name, 'time': elapsed, 'accuracy': acc, 'error': None}
            )
        except Exception as e:
            elapsed = time.time() - start_time
            results.append(
                {'method': name, 'time': elapsed, 'accuracy': None, 'error': str(e)}
            )
            print(f'{name}: 错误: {e}')

    print('\n训练完成:')
    for r in results:
        print(r)


if __name__ == '__main__':

    test_all_classifiers()

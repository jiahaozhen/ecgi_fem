
import time
from knn_V1_V6_17 import knn_classifier
from lda_V1_V6_17 import lda_classifier
from lgb_V1_V6_17 import lgb_classifier
from lmt_V1_V6_17 import lmt_classifier
from adaboost_lmt_V1_V6_17 import adaboost_lmt_classifier
from bayes_V1_V6_17 import bayes_classifier
from xgb_V1_V6_17 import xgb_classifier
from utils.machine_learning_tools import load_dataset, split_dataset

data_dir = 'machine_learning/data/dataset/d_V1_V6_dataset'
X, y = load_dataset(data_dir)
X_train, X_test, y_train, y_test = split_dataset(X, y)

methods = [
    ('KNN', knn_classifier),
    ('LDA', lda_classifier),
    ('LightGBM', lgb_classifier),
    ('LogisticRegression', lmt_classifier),
    ('XGBoost', xgb_classifier)
]

results = []

for name, func in methods:
    print(f'\n训练 {name}...')
    start_time = time.time()
    try:
        clf = func(X_train, y_train)
        elapsed = time.time() - start_time
        print(f'{name}: 训练时间 = {elapsed:.4f}s')
        # 评估模型并记录准确度
        print(f'{name} 测试结果:')
        from sklearn.metrics import accuracy_score
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print('Accuracy:', acc)
        results.append({'method': name, 'time': elapsed, 'accuracy': acc, 'error': None})
    except Exception as e:
        elapsed = time.time() - start_time
        results.append({'method': name, 'time': elapsed, 'accuracy': None, 'error': str(e)})
        print(f'{name}: 错误: {e}')

print('\n训练完成:')
for r in results:
    print(r)

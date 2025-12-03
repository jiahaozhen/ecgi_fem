import xgboost as xgb
from utils.machine_learning_tools import load_dataset, split_dataset, evaluate_model


def xgb_classifier(X_train, y_train):
    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=17,
        learning_rate=0.05,
        n_estimators=100,
        max_depth=6,
        verbosity=0,
        tree_method='auto',
        eval_metric='mlogloss',
    )
    clf.fit(X_train, y_train)
    return clf


if __name__ == '__main__':
    data_dir = 'machine_learning/data/dataset/d64_standard_dataset'
    X, y = load_dataset(data_dir)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    clf = xgb_classifier(X_train, y_train)
    evaluate_model(clf, X_test, y_test)

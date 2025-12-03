from sklearn.linear_model import SGDClassifier
from utils.machine_learning_tools import load_dataset, split_dataset, evaluate_model


def sgd_classifier(X_train, y_train):
    clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    clf.fit(X_train, y_train)
    return clf


if __name__ == '__main__':
    data_dir = 'machine_learning/data/dataset/d6_standard_dataset'
    X, y = load_dataset(data_dir)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    clf = sgd_classifier(X_train, y_train)
    evaluate_model(clf, X_test, y_test)

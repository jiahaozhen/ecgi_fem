from sklearn.neighbors import KNeighborsClassifier
from utils.machine_learning_tools import load_dataset, split_dataset, evaluate_model


def knn_classifier(X_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    return clf


if __name__ == '__main__':
    data_dir = 'machine_learning/data/dataset/d300_standard_dataset'
    X, y = load_dataset(data_dir)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    clf = knn_classifier(X_train, y_train)
    evaluate_model(clf, X_test, y_test)



from sklearn.linear_model import LogisticRegression
from utils.machine_learning_tools import load_dataset, split_dataset, evaluate_model

def lmt_classifier(X_train, y_train):
	clf = LogisticRegression(max_iter=1000, solver='lbfgs')
	clf.fit(X_train, y_train)
	return clf

if __name__ == '__main__':
	data_dir = 'machine_learning/data/dataset/d_V1_V6_dataset'
	X, y = load_dataset(data_dir)
	X_train, X_test, y_train, y_test = split_dataset(X, y)
	clf = lmt_classifier(X_train, y_train)
	evaluate_model(clf, X_test, y_test)

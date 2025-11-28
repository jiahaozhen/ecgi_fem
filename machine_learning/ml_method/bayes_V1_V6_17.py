

from sklearn.naive_bayes import GaussianNB
from utils.machine_learning_tools import load_dataset, split_dataset, evaluate_model

def bayes_classifier(X_train, y_train):
	clf = GaussianNB()
	clf.fit(X_train, y_train)
	return clf

if __name__ == '__main__':
	data_dir = 'machine_learning/data/dataset/d_V1_V6_dataset'
	X, y = load_dataset(data_dir)
	X_train, X_test, y_train, y_test = split_dataset(X, y)
	clf = bayes_classifier(X_train, y_train)
	evaluate_model(clf, X_test, y_test)